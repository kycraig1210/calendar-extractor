from __future__ import annotations
import os, io, re, hashlib
from datetime import datetime, timedelta
from secrets import token_urlsafe
from typing import Optional, List
from zoneinfo import ZoneInfo
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from itertools import groupby
from calendar import month_name

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Boolean, DateTime, ForeignKey, Text, Float

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from icalendar import Calendar, Event as ICalEvent

# Optional: for better PDF text extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# --- Point pytesseract to your install (Windows) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Storage ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}"

# --- DB Models ---
class Base(DeclarativeBase): 
    pass

class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="Source")
    tz: Mapped[str] = mapped_column(String(64), default="America/New_York")
    default_location: Mapped[Optional[str]] = mapped_column(String(256))
    file_path: Mapped[str] = mapped_column(String(512))
    last_hash: Mapped[Optional[str]] = mapped_column(String(64))
    feed_token: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    events: Mapped[List["EventDraft"]] = relationship(back_populates="source", cascade="all,delete")

class EventDraft(Base):
    __tablename__ = "event_drafts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    title: Mapped[str] = mapped_column(String(256))
    start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    all_day: Mapped[bool] = mapped_column(Boolean, default=False)
    location: Mapped[Optional[str]] = mapped_column(String(256))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    rrule: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    uid: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped["Source"] = relationship(back_populates="events")

# --- Database Setup ---
engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Functions ---
def sha256_hex(b: bytes) -> str:
    """Generate SHA256 hash of bytes"""
    return hashlib.sha256(b).hexdigest()

def stable_uid(title: str, start: Optional[datetime], end: Optional[datetime], location: Optional[str]) -> str:
    """Generate stable unique ID for an event"""
    payload = "|".join([
        title or "", 
        start.isoformat() if start else "", 
        end.isoformat() if end else "", 
        location or ""
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

def to_tz(dt: Optional[datetime], tz: str) -> Optional[datetime]:
    """Convert datetime to specified timezone"""
    if not dt: 
        return None
    if dt.tzinfo is None:  # assume local tz
        return dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(ZoneInfo(tz))

# --- OCR and Text Extraction ---
def pdf_to_images(pdf_bytes: bytes):
    """Convert PDF pages to PIL images"""
    with io.BytesIO(pdf_bytes) as f:
        doc = pdfium.PdfDocument(f)
        for i in range(len(doc)):
            yield doc[i].render(scale=2).to_pil().convert("RGB")

def extract_from_right_column_enhanced(img: Image.Image) -> str:
    """
    Enhanced extraction from right column with better preprocessing
    for colored backgrounds
    """
    w, h = img.size
    
    # Try multiple crop positions to catch the dates
    results = []
    
    # Try different column positions (right 50%, 40%, 30%)
    for crop_ratio in [0.5, 0.6, 0.7]:
        left = int(w * crop_ratio)
        cropped = img.crop((left, 0, w, h))
        
        # Convert to grayscale
        gray = ImageOps.grayscale(cropped)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(3.0)
        
        # Binarize to remove colored background
        gray = gray.point(lambda p: 255 if p > 180 else 0)
        
        # Clean up
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        gray = gray.filter(ImageFilter.SHARPEN)
        
        # Upscale
        gray = gray.resize((gray.width * 3, gray.height * 3), Image.LANCZOS)
        
        # OCR with specific config for date lists
        cfg = (
            r'--oem 1 --psm 4 -l eng '  # PSM 4 is good for columns
            r'-c user_defined_dpi=300 '
            r'-c preserve_interword_spaces=1'
        )
        
        txt = pytesseract.image_to_string(gray, config=cfg)
        if txt.strip():
            results.append(txt)
    
    return "\n".join(results)

def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extract text from PDF, trying multiple methods.
    Returns (text, method) where method is 'extracted' or 'ocr'
    """
    # Try PyMuPDF first if available
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            if len(text.strip()) > 100:
                return text, "extracted"
        except:
            pass
    
    # Try pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
            if len(text.strip()) > 100:
                return text, "extracted"
    except:
        pass
    
    # Fallback to OCR
    ocr_pages = []
    for pil in pdf_to_images(pdf_bytes):
        # OCR full page
        full_cfg = r'--oem 1 --psm 6 -l eng -c user_defined_dpi=300'
        full_txt = pytesseract.image_to_string(pil, config=full_cfg)
        # OCR right column for dates
        right_txt = extract_from_right_column_enhanced(pil)
        ocr_pages.append(full_txt + "\n" + right_txt)
    
    return "\n".join(ocr_pages), "ocr"

def extract_text_from_image(img_bytes: bytes) -> tuple[str, str]:
    """Extract text from image using multiple OCR strategies"""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    all_text = []
    
    # Strategy 1: Direct OCR on original
    try:
        cfg = r'--oem 1 --psm 6 -l eng'
        text1 = pytesseract.image_to_string(im, config=cfg)
        if text1.strip():
            all_text.append(text1)
    except:
        pass
    
    # Strategy 2: Grayscale with high contrast
    gray = ImageOps.grayscale(im)
    enhancer = ImageEnhance.Contrast(gray)
    high_contrast = enhancer.enhance(3.0)
    
    # Multiple threshold values for different backgrounds
    for threshold in [180, 150, 120, 200]:
        binary = high_contrast.point(lambda p: 255 if p > threshold else 0)
        binary = binary.filter(ImageFilter.MedianFilter(size=3))
        
        # Try different PSM modes
        for psm in [6, 4, 11]:  # 6=uniform block, 4=column, 11=sparse text
            cfg = f'--oem 1 --psm {psm} -l eng'
            try:
                text = pytesseract.image_to_string(binary, config=cfg)
                if text.strip() and len(text) > 20:
                    all_text.append(text)
            except:
                continue
    
    # Strategy 3: Invert colors (for light text on dark background)
    inverted = ImageOps.invert(gray)
    inverted = inverted.point(lambda p: 255 if p > 150 else 0)
    try:
        cfg = r'--oem 1 --psm 6 -l eng'
        text_inv = pytesseract.image_to_string(inverted, config=cfg)
        if text_inv.strip():
            all_text.append(text_inv)
    except:
        pass
    
    # Combine all extracted text
    combined_text = "\n".join(all_text)
    
    # Also try column extraction
    right_text = extract_from_right_column_enhanced(im)
    
    return combined_text + "\n" + right_text, "ocr"

def extract_text(pdf_bytes: bytes | None, img_bytes: bytes | None) -> tuple[str, str]:
    """
    Main extraction function.
    Returns (text, method) where method is 'extracted' or 'ocr'
    """
    if pdf_bytes:
        return extract_text_from_pdf(pdf_bytes)
    elif img_bytes:
        return extract_text_from_image(img_bytes)
    else:
        return "", "none"

# --- Event Parsing ---
def canon_title(raw: str) -> str:
    """
    Normalize noisy OCR titles to clean, canonical labels.
    """
    # Clean up: remove special chars, lowercase, collapse spaces
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", raw).lower()
    s = re.sub(r"\s+", " ", s).strip()
    jam = s.replace(" ", "")  # for matching without spaces
    
    # Fix common OCR errors
    replacements = {
        "studenv": "student",
        "holida": "holiday",
        "inclement": "inclement",
        "makeup": "makeup",
        "make up": "makeup",
        "workday": "workday",
        "closure": "closure",
        "teacherworkday": "teacherworkday"
    }
    
    for old, new in replacements.items():
        jam = jam.replace(old, new)
    
    # Canonical mappings (order matters: more specific first)
    mappings = [
        (["firstdayofschool", "firstday", "startofschool"], "First Day of School"),
        (["lastdayofschool", "lastday", "endofschool"], "Last Day of School"),
        (["studentstaffholiday", "studentholiday", "staffholiday"], "Student/Staff Holiday"),
        (["thanksgiving", "thanksgivingbreak"], "Thanksgiving Break"),
        (["springbreak"], "Spring Break"),
        (["winterbreak", "winterholiday"], "Winter Break"),
        (["laborday"], "Labor Day"),
        (["memorialday"], "Memorial Day"),
        (["mlkday", "martinlutherking"], "MLK Day"),
        (["presidentsday", "presidentsday"], "Presidents Day"),
        (["teacherworkday", "schoolclosure", "makeupday"], "Teacher Work Day"),
        (["inclementweatherday", "inclementweather", "weatherday"], "Inclement Weather Day"),
        (["professionalday", "professionaldays", "profday"], "Professional Day"),
        (["earlyrelease", "earlydismissal"], "Early Release"),
        (["parentteacherconference", "conferences"], "Parent-Teacher Conference"),
    ]
    
    for keywords, canonical in mappings:
        if any(k in jam for k in keywords):
            return canonical
    
    # Fallback: title case the original, limit length
    words = s.split()[:6]  # Max 6 words
    return " ".join(w.capitalize() for w in words) if words else "School Event"

def extract_title_from_segment(segment: str) -> str:
    """Extract clean title from text segment after date"""
    # Remove leading separators and whitespace
    segment = re.sub(r'^[\s\-–—•:|\\/]+', '', segment)
    
    # Take first line only
    lines = segment.split('\n')
    title = lines[0] if lines else ''
    
    # Remove trailing junk
    title = re.sub(r'[\d\s\-–—•:|\\/]+$', '', title)
    
    return title.strip()

def calculate_confidence(event: dict, extraction_method: str) -> float:
    """Calculate confidence score for extracted event"""
    confidence = 0.5  # Base confidence
    
    # Boost for extraction method
    if extraction_method == "extracted":
        confidence += 0.2
    
    # Boost for known event types
    known_events = [
        "First Day of School", "Last Day of School", 
        "Student/Staff Holiday", "Professional Day",
        "Thanksgiving Break", "Spring Break", "Winter Break",
        "Labor Day", "Memorial Day", "MLK Day"
    ]
    if event['title'] in known_events:
        confidence += 0.2
    
    # Boost for clean title (no weird characters)
    if not re.search(r'[?!@#$%^&*()]', event['title']):
        confidence += 0.1
    
    # Check date validity
    if event['start']:
        year = event['start'].year
        if year in [2025, 2026]:  # Expected academic years
            confidence += 0.1
        if 8 <= event['start'].month <= 12 or 1 <= event['start'].month <= 6:
            confidence += 0.05  # Within typical school year
    
    return min(confidence, 1.0)

def parse_events_from_text(text: str, tz: str, default_location: Optional[str], extraction_method: str = "ocr") -> list[dict]:
    """
    Parse events from extracted text using multiple strategies.
    Enhanced to handle email/web content with dates in sentences.
    """
    events: list[dict] = []
    
    # Detect current year context
    import re
    from dateutil import parser as date_parser
    
    years = re.findall(r"(20\d{2})", text)
    current_year = int(years[0]) if years else datetime.now().year
    
    # Strategy 1: Look for explicit date patterns in sentences
    # "Season begins September 6th", "starts on October 1", etc.
    sentence_patterns = [
        r"(?:begins?|starts?|starting)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
        r"(?:ends?|ending|concludes?)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
        r"(?:from|between)\s+([A-Za-z]+\s+\d{1,2})\s+(?:to|through|until|-)\s+([A-Za-z]+\s+\d{1,2})",
        r"([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\s*[-–]\s*([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
        r"(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
    ]
    
    seen_events = set()
    
    for pattern in sentence_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Parse the date(s)
                date_str = match.group(1)
                parsed_date = date_parser.parse(f"{date_str} {current_year}", fuzzy=True)
                
                # Determine event type from context
                context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                
                title = "Event"
                if re.search(r"season\s+(?:begins?|starts?)", context, re.I):
                    title = "Season Begins"
                elif re.search(r"season\s+ends?", context, re.I):
                    title = "Season Ends"
                elif re.search(r"no\s+(?:games?|practice|school)", context, re.I):
                    title = "No Games/Practice"
                elif re.search(r"practice", context, re.I):
                    title = "Practice"
                elif re.search(r"games?", context, re.I):
                    title = "Game"
                elif re.search(r"first\s+day", context, re.I):
                    title = "First Day"
                elif re.search(r"last\s+day", context, re.I):
                    title = "Last Day"
                elif re.search(r"holiday|break|vacation", context, re.I):
                    title = "Holiday/Break"
                elif re.search(r"meeting|conference", context, re.I):
                    title = "Meeting/Conference"
                
                # Handle date ranges if second date exists
                end_date = None
                if match.lastindex and match.lastindex >= 2:
                    try:
                        date_str2 = match.group(2)
                        end_date = date_parser.parse(f"{date_str2} {current_year}", fuzzy=True)
                    except:
                        pass
                
                # Create event
                start = parsed_date.replace(hour=0, minute=0, second=0, tzinfo=ZoneInfo(tz))
                end = end_date.replace(hour=0, minute=0, second=0, tzinfo=ZoneInfo(tz)) + timedelta(days=1) if end_date else start + timedelta(days=1)
                
                event_key = (title, start.date(), end.date())
                if event_key not in seen_events:
                    event = {
                        "title": title,
                        "start": start,
                        "end": end,
                        "all_day": True,
                        "location": default_location,
                        "notes": None,
                        "rrule": None,
                        "confidence": calculate_confidence({"title": title, "start": start}, extraction_method)
                    }
                    events.append(event)
                    seen_events.add(event_key)
            except:
                continue
    
    # Strategy 2: Original month + day pattern for calendar formats
    date_pat = re.compile(
        r"(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\.?\s*"
        r"(?P<d1>\d{1,2})"
        r"(?:\s*[-–—]\s*(?P<d2>\d{1,2}))?",
        re.IGNORECASE
    )
    
    MONTHS = {
        "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3,
        "APR": 4, "APRIL": 4, "MAY": 5, "JUN": 6, "JUNE": 6,
        "JUL": 7, "JULY": 7, "AUG": 8, "AUGUST": 8, "SEP": 9, "SEPT": 9, 
        "SEPTEMBER": 9, "OCT": 10, "OCTOBER": 10, "NOV": 11, "NOVEMBER": 11,
        "DEC": 12, "DECEMBER": 12,
    }
    
    tokens = list(date_pat.finditer(text))
    
    for i, match in enumerate(tokens):
        mon_text = match.group("mon").upper().replace(".", "")
        if mon_text not in MONTHS:
            continue
            
        month = MONTHS[mon_text]
        day1 = int(match.group("d1"))
        day2_match = match.group("d2")
        
        # Extract context for title
        seg_start = match.end()
        seg_end = tokens[i + 1].start() if i + 1 < len(tokens) else min(match.end() + 100, len(text))
        raw_segment = text[seg_start:seg_end]
        
        title_text = extract_title_from_segment(raw_segment)
        title = canon_title(title_text) if title_text else "School Event"
        
        # Determine year based on academic calendar
        year = current_year if month >= 8 else current_year + 1
        
        try:
            start = datetime(year, month, day1, 0, 0, 0, tzinfo=ZoneInfo(tz))
            
            if day2_match:
                day2 = int(day2_match)
                end = datetime(year, month, day2, 0, 0, 0, tzinfo=ZoneInfo(tz)) + timedelta(days=1)
            else:
                end = start + timedelta(days=1)
            
            event_key = (title, start.date(), end.date())
            if event_key not in seen_events:
                event = {
                    "title": title,
                    "start": start,
                    "end": end,
                    "all_day": True,
                    "location": default_location,
                    "notes": None,
                    "rrule": None,
                    "confidence": calculate_confidence({"title": title, "start": start}, extraction_method)
                }
                events.append(event)
                seen_events.add(event_key)
                
        except ValueError:
            continue
    
    # Strategy 3: Look for recurring patterns (weekly practices, games)
    # "Every Tuesday", "Thursdays at 5pm", etc.
    recurring_pattern = re.compile(
        r"(?:every|each)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
        r"(?:\s+at\s+(\d{1,2}(?::\d{2})?\s*[ap]m))?",
        re.IGNORECASE
    )
    
    for match in recurring_pattern.finditer(text):
        day_name = match.group(1)
        time_str = match.group(2) if match.group(2) else None
        
        # This would need RRULE support for recurring events
        # For now, just note it exists
        
    return events

# --- ICS Calendar Builder ---
def build_ics(approved: list[EventDraft]) -> bytes:
    """Build ICS calendar from approved events"""
    cal = Calendar()
    cal.add("prodid", "-//School Calendar AI//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", "School Calendar")
    
    for e in approved:
        ve = ICalEvent()
        ve.add("uid", e.uid)
        ve.add("summary", e.title)
        
        if e.all_day:
            ve.add("dtstart", e.start.date() if e.start else datetime.now().date())
            ve.add("dtend", e.end.date() if e.end else datetime.now().date())
        else:
            ve.add("dtstart", e.start)
            if e.end:
                ve.add("dtend", e.end)
        
        if e.location:
            ve.add("location", e.location)
        if e.notes:
            ve.add("description", e.notes)
        if e.rrule:
            ve.add("rrule", e.rrule)
        
        ve.add("dtstamp", datetime.now(ZoneInfo("UTC")))
        cal.add_component(ve)
    
    return cal.to_ical()

# --- FastAPI Application ---
app = FastAPI(title="School Calendar Extraction API")

@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Simple home page with text input form"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calendar Text Extractor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #2d3748;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #718096;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                border-bottom: 2px solid #e2e8f0;
            }
            .tab {
                padding: 10px 20px;
                background: none;
                border: none;
                color: #718096;
                font-size: 16px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.2s;
            }
            .tab.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            textarea {
                width: 100%;
                min-height: 300px;
                padding: 15px;
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                font-size: 15px;
                resize: vertical;
                font-family: inherit;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            input[type="text"], input[type="file"], input[type="date"], select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 15px;
                margin-bottom: 15px;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #4a5568;
                font-weight: 600;
            }
            .btn {
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-primary {
                background: #667eea;
                color: white;
            }
            .btn-primary:hover {
                background: #5a67d8;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            .examples {
                margin-top: 20px;
                padding: 20px;
                background: #f7fafc;
                border-radius: 10px;
            }
            .example {
                margin-bottom: 10px;
                color: #4a5568;
            }
            .example strong {
                color: #2d3748;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📅 Calendar Text Extractor</h1>
            <p class="subtitle">Extract dates from emails, websites, or any text</p>
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab(event, 'paste')">📝 Paste Text</button>
                <button class="tab" onclick="switchTab(event, 'upload')">📤 Upload File</button>
            </div>
            
            <!-- Paste Text Tab -->
            <div id="paste-tab" class="tab-content active">
                <form id="text-form">
                    <div class="form-group">
                        <label for="text">Paste your calendar text, email, or schedule:</label>
                        <textarea name="text" id="text" placeholder="Example:
The season will begin on September 6th and end on November 1st.

Practice Schedule:
- Tuesdays and Thursdays at 5:30pm
- No practice on October 10th

Important Dates:
First Day of School: August 13
Labor Day Holiday: September 1
Parent-Teacher Conference: October 15-16
Winter Break: December 19 - January 2" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="name">Calendar Name:</label>
                        <input type="text" name="name" id="name" value="My Calendar">
                    </div>
                    
                    <div class="form-group">
                        <label for="tz">Time Zone:</label>
                        <select name="tz" id="tz">
                            <option value="America/New_York">Eastern Time</option>
                            <option value="America/Chicago">Central Time</option>
                            <option value="America/Denver">Mountain Time</option>
                            <option value="America/Los_Angeles">Pacific Time</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="default_location">Default Location (optional):</label>
                        <input type="text" name="default_location" id="default_location" placeholder="e.g., Lincoln Elementary School">
                    </div>
                    
                    <button type="button" onclick="submitText()" class="btn btn-primary">Extract Dates →</button>
                </form>
                
                <div class="examples">
                    <h3>✨ What works best:</h3>
                    <div class="example">• <strong>Emails:</strong> "The meeting is scheduled for March 15th at 7pm"</div>
                    <div class="example">• <strong>Lists:</strong> "Aug 13 - First Day of School"</div>
                    <div class="example">• <strong>Sentences:</strong> "Season runs from September 6 to November 1"</div>
                </div>
            </div>
            
            <!-- Upload File Tab -->
            <div id="upload-tab" class="tab-content">
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Upload Calendar Image or PDF:</label>
                        <input type="file" name="file" id="file" accept="image/*,.pdf" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="upload-name">Calendar Name:</label>
                        <input type="text" name="name" id="upload-name" value="School Calendar">
                    </div>
                    
                    <div class="form-group">
                        <label for="upload-tz">Time Zone:</label>
                        <select name="tz" id="upload-tz">
                            <option value="America/New_York">Eastern Time</option>
                            <option value="America/Chicago">Central Time</option>
                            <option value="America/Denver">Mountain Time</option>
                            <option value="America/Los_Angeles">Pacific Time</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="upload-location">Default Location (optional):</label>
                        <input type="text" name="default_location" id="upload-location">
                    </div>
                    
                    <button type="button" onclick="submitUpload()" class="btn btn-primary">Upload & Extract →</button>
                </form>
            </div>
        </div>
        
        <script>
            function switchTab(event, tab) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById(tab + '-tab').classList.add('active');
                event.target.classList.add('active');
            }
            
            async function submitText() {
                const formData = new FormData();
                formData.append('text', document.getElementById('text').value);
                formData.append('name', document.getElementById('name').value);
                formData.append('tz', document.getElementById('tz').value);
                formData.append('default_location', document.getElementById('default_location').value);
                
                try {
                    const response = await fetch('/sources/text', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        // Redirect to review page
                        window.location.href = data.review_url;
                    } else {
                        alert('Error extracting dates. Please try again.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error extracting dates. Please try again.');
                }
            }
            
            async function submitUpload() {
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                
                if (!fileInput.files[0]) {
                    alert('Please select a file');
                    return;
                }
                
                formData.append('file', fileInput.files[0]);
                formData.append('name', document.getElementById('upload-name').value);
                formData.append('tz', document.getElementById('upload-tz').value);
                formData.append('default_location', document.getElementById('upload-location').value);
                
                try {
                    const response = await fetch('/sources/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        // Redirect to review page
                        window.location.href = data.review_url;
                    } else {
                        alert('Error uploading file. Please try again.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error uploading file. Please try again.');
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/sources/upload")
async def upload_source(
    name: str = Form("School Calendar"),
    tz: str = Form("America/New_York"),
    default_location: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a calendar file OR paste text content directly"""
    
    if not file and not text_content:
        raise HTTPException(400, "Either file or text_content must be provided")
    
    # Process based on input type
    if file:
        # File upload path (existing logic)
        content = await file.read()
        
        # Save uploaded file
        filename = token_urlsafe(8) + "_" + file.filename
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as f:
            f.write(content)
        
        # Extract text
        is_pdf = file.filename.lower().endswith(".pdf")
        text, method = extract_text(
            pdf_bytes=content if is_pdf else None,
            img_bytes=content if not is_pdf else None
        )
        
        file_hash = sha256_hex(content)
    else:
        # Direct text input path
        text = text_content
        method = "text"
        
        # Save text as file for consistency
        filename = token_urlsafe(8) + "_text_input.txt"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        
        file_hash = sha256_hex(text.encode())
    
    # Create source record
    src = Source(
        name=name,
        tz=tz,
        default_location=default_location,
        file_path=path,
        last_hash=file_hash,
        feed_token=token_urlsafe(24)
    )
    db.add(src)
    db.commit()
    db.refresh(src)
    
    # Parse events
    events = parse_events_from_text(text, tz, default_location, method)
    
    # Save events to database
    for ev in events:
        uid = stable_uid(ev["title"], ev["start"], ev["end"], ev["location"])
        db.add(EventDraft(source_id=src.id, uid=uid, approved=False, **ev))
    db.commit()
    
    # Return info
    return {
        "source_id": src.id,
        "events_created": len(events),
        "extraction_method": method,
        "feed_url": f"/feeds/{src.feed_token}.ics",
        "review_url": f"/sources/{src.id}/review"
    }

@app.post("/sources/text")
async def create_from_text(
    text: str = Form(...),
    name: str = Form("Calendar from Text"),
    tz: str = Form("America/New_York"),
    default_location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Simple endpoint for text-only input (for web forms)"""
    return await upload_source(
        name=name,
        tz=tz,
        default_location=default_location,
        file=None,
        text_content=text,
        db=db
    )

@app.post("/events/{event_id}/update")
def update_event(
    event_id: int,
    title: Optional[str] = Form(None),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Update an existing event"""
    event = db.get(EventDraft, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    
    if title:
        event.title = title
    if start_date:
        event.start = datetime.fromisoformat(start_date).replace(tzinfo=ZoneInfo(event.source.tz))
    if end_date:
        event.end = datetime.fromisoformat(end_date).replace(tzinfo=ZoneInfo(event.source.tz))
    if location:
        event.location = location
    if notes:
        event.notes = notes
    
    db.commit()
    
    return {"id": event_id, "updated": True}

@app.post("/sources/{source_id}/add_event")
def add_manual_event(
    source_id: int,
    title: str = Form(...),
    start_date: str = Form(...),
    end_date: Optional[str] = Form(None),
    all_day: bool = Form(True),
    location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    recurring: Optional[str] = Form(None),  # "weekly", "monthly", etc.
    recur_until: Optional[str] = Form(None),  # End date for recurring events
    db: Session = Depends(get_db)
):
    """Manually add an event to a source"""
    source = db.get(Source, source_id)
    if not source:
        raise HTTPException(404, "Source not found")
    
    # Parse dates
    start = datetime.fromisoformat(start_date).replace(tzinfo=ZoneInfo(source.tz))
    end = datetime.fromisoformat(end_date).replace(tzinfo=ZoneInfo(source.tz)) if end_date else start + timedelta(days=1)
    
    # Handle recurring events
    events_to_add = []
    
    if recurring and recur_until:
        # Generate recurring events
        recur_end = datetime.fromisoformat(recur_until).replace(tzinfo=ZoneInfo(source.tz))
        current_date = start
        
        while current_date <= recur_end:
            event_start = current_date
            event_end = event_start + (end - start) if end else event_start + timedelta(days=1)
            
            events_to_add.append({
                "title": title,
                "start": event_start,
                "end": event_end,
                "all_day": all_day,
                "location": location or source.default_location,
                "notes": notes,
                "confidence": 1.0,  # Manual events have high confidence
                "approved": True  # Auto-approve manual events
            })
            
            # Move to next occurrence
            if recurring == "daily":
                current_date += timedelta(days=1)
            elif recurring == "weekly":
                current_date += timedelta(weeks=1)
            elif recurring == "biweekly":
                current_date += timedelta(weeks=2)
            elif recurring == "monthly":
                # Add a month (handle varying days)
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            else:
                break  # Unknown recurring type
    else:
        # Single event
        events_to_add.append({
            "title": title,
            "start": start,
            "end": end,
            "all_day": all_day,
            "location": location or source.default_location,
            "notes": notes,
            "confidence": 1.0,
            "approved": True
        })
    
    # Add all events to database
    added_count = 0
    for ev in events_to_add:
        uid = stable_uid(ev["title"], ev["start"], ev["end"], ev["location"])
        # Check for duplicates
        existing = db.query(EventDraft).filter(
            EventDraft.source_id == source_id,
            EventDraft.uid == uid
        ).first()
        
        if not existing:
            db.add(EventDraft(source_id=source_id, uid=uid, rrule=None, **ev))
            added_count += 1
    
    db.commit()
    
    return {
        "source_id": source_id,
        "events_added": added_count,
        "recurring": recurring is not None
    }

@app.get("/sources/{source_id}/events")
def list_events(source_id: int, db: Session = Depends(get_db)):
    """List all events for a source"""
    events = db.scalars(
        select(EventDraft)
        .where(EventDraft.source_id == source_id)
        .order_by(EventDraft.start)
    ).all()
    
    return [
        {
            "id": e.id,
            "title": e.title,
            "start": e.start.isoformat() if e.start else None,
            "end": e.end.isoformat() if e.end else None,
            "all_day": e.all_day,
            "location": e.location,
            "notes": e.notes,
            "confidence": e.confidence,
            "approved": e.approved,
            "uid": e.uid
        }
        for e in events
    ]

@app.post("/events/{event_id}/toggle")
def toggle_event_approval(event_id: int, db: Session = Depends(get_db)):
    """Toggle approval status of an event"""
    event = db.get(EventDraft, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    
    event.approved = not event.approved
    db.commit()
    
    return {"id": event_id, "approved": event.approved}

@app.post("/sources/{source_id}/approve_all")
def approve_all_events(source_id: int, db: Session = Depends(get_db)):
    """Approve all events for a source"""
    events = db.scalars(
        select(EventDraft).where(EventDraft.source_id == source_id)
    ).all()
    
    for event in events:
        event.approved = True
    
    db.commit()
    
    return {"approved_count": len(events)}

@app.get("/feeds/{feed_token}.ics")
def serve_feed(feed_token: str, db: Session = Depends(get_db)):
    """Serve ICS calendar feed"""
    source = db.scalar(
        select(Source).where(Source.feed_token == feed_token)
    )
    
    if not source:
        raise HTTPException(404, "Unknown feed token")
    
    approved_events = db.scalars(
        select(EventDraft)
        .where(EventDraft.source_id == source.id, EventDraft.approved == True)
        .order_by(EventDraft.start)
    ).all()
    
    ics_content = build_ics(approved_events)
    
    return Response(
        content=ics_content,
        media_type="text/calendar",
        headers={
            "Content-Disposition": f'inline; filename="{source.name}.ics"'
        }
    )

@app.get("/sources/{source_id}/review", response_class=HTMLResponse)
async def review_page(source_id: int, db: Session = Depends(get_db)):
    """Interactive review page for extracted events"""
    source = db.get(Source, source_id)
    if not source:
        raise HTTPException(404, "Source not found")
    
    events = db.scalars(
        select(EventDraft)
        .where(EventDraft.source_id == source_id)
        .order_by(EventDraft.start)
    ).all()
    
    # Build HTML page - Fixed the dictionary initialization syntax
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Review: {source.name}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            h1 {{ 
                font-size: 2em;
                margin-bottom: 10px;
            }}
            .subtitle {{
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .controls {{
                background: #f8f9fa;
                padding: 20px 30px;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }}
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            .btn-primary {{
                background: #667eea;
                color: white;
            }}
            .btn-primary:hover {{
                background: #5a67d8;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .btn-success {{
                background: #48bb78;
                color: white;
            }}
            .btn-success:hover {{
                background: #38a169;
                transform: translateY(-1px);
            }}
            .stats {{
                margin-left: auto;
                color: #6c757d;
                font-size: 14px;
            }}
            .events-container {{
                padding: 30px;
                max-height: 600px;
                overflow-y: auto;
            }}
            .month-group {{
                margin-bottom: 30px;
            }}
            .month-header {{
                font-size: 1.2em;
                color: #667eea;
                font-weight: 600;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e9ecef;
            }}
            .event {{
                background: white;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 12px;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .event:hover {{
                border-color: #667eea;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                transform: translateX(5px);
            }}
            .event.approved {{
                background: #f0fff4;
                border-color: #48bb78;
            }}
            .event.low-confidence {{
                background: #fffaf0;
                border-color: #ed8936;
            }}
            .event-checkbox {{
                width: 20px;
                height: 20px;
                cursor: pointer;
            }}
            .event-content {{
                flex: 1;
            }}
            .event-title {{
                font-weight: 600;
                color: #2d3748;
                font-size: 1.05em;
                margin-bottom: 4px;
            }}
            .event-date {{
                color: #718096;
                font-size: 0.95em;
            }}
            .event-meta {{
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 0.85em;
                color: #a0aec0;
                margin-top: 5px;
            }}
            .confidence {{
                padding: 2px 8px;
                border-radius: 12px;
                background: #edf2f7;
                font-weight: 600;
            }}
            .confidence.high {{ 
                background: #c6f6d5; 
                color: #276749;
            }}
            .confidence.medium {{ 
                background: #fed7aa; 
                color: #7c2d12;
            }}
            .confidence.low {{ 
                background: #fed7d7; 
                color: #742a2a;
            }}
            .modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }}
            .modal.show {{
                display: flex;
            }}
            .modal-content {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                max-width: 500px;
                text-align: center;
            }}
            .modal h2 {{
                margin-bottom: 15px;
                color: #2d3748;
            }}
            .modal p {{
                color: #718096;
                margin-bottom: 20px;
                line-height: 1.6;
            }}
            .feed-url {{
                background: #f7fafc;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 20px;
                word-break: break-all;
                font-family: monospace;
                font-size: 0.9em;
            }}
            .copy-btn {{
                background: #667eea;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            }}
            .copy-btn:hover {{
                background: #5a67d8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📅 {source.name}</h1>
                <div class="subtitle">Review and approve extracted calendar events</div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="toggleAll()">
                    <span>☑️</span> Toggle All
                </button>
                <button class="btn btn-success" onclick="showFeedModal()">
                    <span>📤</span> Get Calendar Feed
                </button>
                <div class="stats">
                    <span id="approved-count">0</span> / {len(events)} approved
                </div>
            </div>
            
            <div class="events-container">
    """
    
    # Group events by month - Fixed dictionary initialization
    events_by_month = dict()
    for event in events:
        if event.start:
            key = (event.start.year, event.start.month)
            if key not in events_by_month:
                events_by_month[key] = []
            events_by_month[key].append(event)
    
    # Sort by date
    for (year, month) in sorted(events_by_month.keys()):
        month_events = events_by_month[(year, month)]
        
        html += f"""
            <div class="month-group">
                <div class="month-header">{month_name[month]} {year}</div>
        """
        
        for event in sorted(month_events, key=lambda e: e.start or datetime.min):
            # Determine confidence level
            conf_class = "high"
            if event.confidence:
                if event.confidence < 0.6:
                    conf_class = "low"
                elif event.confidence < 0.8:
                    conf_class = "medium"
            
            approved_class = "approved" if event.approved else ""
            low_conf_class = "low-confidence" if event.confidence and event.confidence < 0.7 else ""
            
            # Format date
            date_str = ""
            if event.start:
                date_str = event.start.strftime('%B %d')
                if event.end and event.end != event.start + timedelta(days=1):
                    end_date = (event.end - timedelta(days=1)).strftime('%d') if event.end.month == event.start.month else (event.end - timedelta(days=1)).strftime('%B %d')
                    date_str += f" - {end_date}"
            
            html += f"""
                <div class="event {approved_class} {low_conf_class}" data-id="{event.id}">
                    <input type="checkbox" 
                           class="event-checkbox" 
                           data-id="{event.id}"
                           {'checked' if event.approved else ''}
                           onchange="toggleEvent({event.id})">
                    <div class="event-content">
                        <div class="event-title">{event.title}</div>
                        <div class="event-date">{date_str}</div>
                        <div class="event-meta">
                            <span class="confidence {conf_class}">
                                {int((event.confidence or 0) * 100)}% confident
                            </span>
                            {f'<span>📍 {event.location}</span>' if event.location else ''}
                        </div>
                    </div>
                </div>
            """
        
        html += "</div>"
    
    # Get the feed URL properly
    feed_url = f"/feeds/{source.feed_token}.ics"
    
    html += f"""
            </div>
        </div>
        
        <div class="modal" id="feedModal">
            <div class="modal-content">
                <h2>🎉 Your Calendar Feed is Ready!</h2>
                <p>Copy this URL and add it to Google Calendar, Apple Calendar, or Outlook as a subscription:</p>
                <div class="feed-url" id="feedUrl">{feed_url}</div>
                <button class="copy-btn" onclick="copyFeedUrl()">📋 Copy URL</button>
                <button class="btn" onclick="closeFeedModal()" style="margin-left: 10px;">Close</button>
            </div>
        </div>
        
        <script>
            const feedUrl = window.location.origin + '/feeds/{source.feed_token}.ics';
            
            function updateCount() {{
                const checked = document.querySelectorAll('.event-checkbox:checked').length;
                document.getElementById('approved-count').textContent = checked;
            }}
            
            async function toggleEvent(eventId) {{
                await fetch(`/events/${{eventId}}/toggle`, {{ method: 'POST' }});
                updateCount();
                
                const checkbox = document.querySelector(`.event-checkbox[data-id="${{eventId}}"]`);
                const event = document.querySelector(`.event[data-id="${{eventId}}"]`);
                
                if (checkbox.checked) {{
                    event.classList.add('approved');
                }} else {{
                    event.classList.remove('approved');
                }}
            }}
            
            function toggleAll() {{
                const checkboxes = document.querySelectorAll('.event-checkbox');
                const allChecked = Array.from(checkboxes).every(cb => cb.checked);
                
                checkboxes.forEach(cb => {{
                    cb.checked = !allChecked;
                    toggleEvent(cb.dataset.id);
                }});
            }}
            
            function showFeedModal() {{
                document.getElementById('feedModal').classList.add('show');
                document.getElementById('feedUrl').textContent = feedUrl;
            }}
            
            function closeFeedModal() {{
                document.getElementById('feedModal').classList.remove('show');
            }}
            
            async function copyFeedUrl() {{
                try {{
                    await navigator.clipboard.writeText(feedUrl);
                    alert('✅ Calendar URL copied! Add it to your calendar app as a subscription.');
                }} catch (err) {{
                    // Fallback for older browsers
                    const textArea = document.createElement("textarea");
                    textArea.value = feedUrl;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    alert('✅ Calendar URL copied!');
                }}
            }}
            
            // Initialize count
            updateCount();
        </script>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)