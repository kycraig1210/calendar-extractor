from __future__ import annotations

import os
import io
import re
import hashlib
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from typing import Optional, List
from zoneinfo import ZoneInfo

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import (
    sessionmaker,
    Session,
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy import Integer, String, Boolean, DateTime, ForeignKey, Text

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import dateparser
from icalendar import Calendar, Event

# ---- Configure pytesseract (Windows default install path) ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Storage / DB ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}"

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
    events: Mapped[List["EventDraft"]] = relationship(
        back_populates="source", cascade="all,delete"
    )

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
    confidence: Mapped[Optional[float]] = mapped_column()
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    uid: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped["Source"] = relationship(back_populates="events")

engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- Helpers ----
MMDD = r"\d{1,2}[/-]\d{1,2}"
DATE_LINE = re.compile(rf"(?P<md>\b{MMDD}\b).*?(?P<title>[^\n]+)", re.I)
TIME_RANGE = re.compile(
    r"(?P<s>\d{1,2}(:\d{2})?\s?(AM|PM)?)\s?[-–—]\s?(?P<e>\d{1,2}(:\d{2})?\s?(AM|PM)?)",
    re.I,
)
ALL_DAY_HINT = re.compile(
    r"\ball[-\s]?day\b|\bno school\b|\bholiday\b|\bearly release\b", re.I
)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_uid(title: str, start: Optional[datetime], end: Optional[datetime], location: Optional[str]) -> str:
    payload = "|".join([title or "", start.isoformat() if start else "", end.isoformat() if end else "", location or ""])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

def pdf_to_images(pdf_bytes: bytes):
    with io.BytesIO(pdf_bytes) as f:
        doc = pdfium.PdfDocument(f)
        for i in range(len(doc)):
            yield doc[i].render(scale=2).to_pil().convert("RGB")

# ---- OCR tuning for the right column (IMPORTANT DATES) ----
def ocr_right_column_from_image(img: Image.Image) -> str:
    w, h = img.size
    # Wider crop to reliably catch the right panel (adjust if needed)
    left = int(w * 0.60)  # try 0.65 or 0.70 if your image differs
    right_col = img.crop((left, 0, w, h))

    g = ImageOps.grayscale(right_col)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.GaussianBlur(0.6))
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    g = g.resize((g.width * 3, g.height * 3))

    bw = g.point(lambda p: 255 if p > 190 else 0)

    cfg = (
        r'--oem 1 --psm 6 -l eng '
        r'-c user_defined_dpi=300 '
        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-–—:/ '
    )
    txt = pytesseract.image_to_string(bw, config=cfg)
    print("\n--- OCR RIGHT COLUMN (first 400 chars) ---\n", txt[:400], "\n--- END ---\n")
    return txt

def extract_text(pdf_bytes: bytes | None, img_bytes: bytes | None) -> str:
    """
    - If PDF contains selectable text, use it.
    - If not, OCR the full page + right column and combine.
    """
    if pdf_bytes:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if text.strip():
            return text

        chunks = []
        for pil in pdf_to_images(pdf_bytes):
            full_cfg = r'--oem 1 --psm 6 -l eng -c user_defined_dpi=300'
            full_txt = pytesseract.image_to_string(pil, config=full_cfg)
            right_txt = ocr_right_column_from_image(pil)
            chunks.append(full_txt + "\n" + right_txt)
        return "\n".join(chunks)

    else:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        full_cfg = r'--oem 1 --psm 6 -l eng -c user_defined_dpi=300'
        full_txt = pytesseract.image_to_string(im, config=full_cfg)
        right_txt = ocr_right_column_from_image(im)
        return full_txt + "\n" + right_txt

# ---- Parser (tolerant to messy OCR) ----
def canon_title(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", raw).lower()
    s = s.replace("  ", " ")
    jam = s.replace(" ", "")

    # Fix common OCR glitches
    jam = (
        jam.replace("studenv", "student")
        .replace("holida", "holiday")
        .replace("inclement", "inclement")
        .replace("makeup", "makeup")
        .replace("workday", "workday")
        .replace("teacherworkday", "teacherworkday")
        .replace("closure", "closure")
        .replace("studentstaffholiday", "student/staﬀholiday")  # normalize
    )

    if any(k in jam for k in ["firstdayofschool", "firstday", "startofschool"]):
        return "First Day of School"
    if any(k in jam for k in ["lastdayofschool", "lastday", "endofschool"]):
        return "Last Day of School"
    if any(k in jam for k in ["student/staffholiday", "studentholiday", "staffholiday", "studentstaffholiday"]):
        return "Student/Staff Holiday"
    if any(k in jam for k in [
        "teacherworkdayschoolclosuremakeupdaystudentholiday",
        "schoolclosuremakeupdaystudentholiday",
        "teacherworkday", "schoolclosure", "makeupday"
    ]):
        return "Teacher Work Day / School Closure Make-up Day / Student Holiday"
    if any(k in jam for k in ["inclementweatherday", "inclementweather"]):
        return "Inclement Weather Day"
    if any(k in jam for k in ["professionalday", "professionaldays"]):
        return "Professional Day"

    words = [w for w in re.sub(r"\s+", " ", s).strip().split(" ") if w]
    return " ".join(w.capitalize() for w in words[:8]) or "School Event"

def parse_events_from_text(text: str, tz: str, default_location: Optional[str]) -> list[dict]:
    events: list[dict] = []

    yrs = re.findall(r"(20\d{2})", text)
    if len(yrs) >= 2:
        start_year = min(int(y) for y in yrs)
    elif len(yrs) == 1:
        start_year = int(yrs[0])
    else:
        start_year = datetime.now().year

    def year_for_month(mon: int) -> int:
        return start_year if mon >= 8 else start_year + 1

    MON = {
        "JAN":1,"JAN.":1,"JANUARY":1,
        "FEB":2,"FEB.":2,"FEBRUARY":2,
        "MAR":3,"MAR.":3,"MARCH":3,
        "APR":4,"APR.":4,"APRIL":4,
        "MAY":5,
        "JUN":6,"JUN.":6,"JUNE":6,
        "JUL":7,"JUL.":7,"JULY":7,
        "AUG":8,"AUG.":8,"AUGUST":8,
        "SEP":9,"SEP.":9,"SEPT":9,"SEPT.":9,"SEPTEMBER":9,
        "OCT":10,"OCT.":10,"OCTOBER":10,
        "NOV":11,"NOV.":11,"NOVEMBER":11,
        "DEC":12,"DEC.":12,"DECEMBER":12,
    }

    full = text

    date_pat = re.compile(
        r"(?P<mon>Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|May|Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:t(?:ember)?)?\.?|Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?)"
        r"[ \t:]*"
        r"(?P<d1>\d{1,2})"
        r"(?:\s*[-–—]\s*(?P<d2>\d{1,2}))?",
        re.IGNORECASE
    )

    tokens = list(date_pat.finditer(full))
    if not tokens:
        # Last resort: try MM/DD Title lines (very basic)
        for raw in full.splitlines():
            m = DATE_LINE.search(raw)
            if not m:
                continue
            md = m.group("md")
            title = canon_title(m.group("title"))
            base = dateparser.parse(f"{md}/{start_year}")
            if not base:
                continue
            start = base.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            events.append({
                "title": title, "start": start, "end": end,
                "all_day": True, "location": default_location,
                "notes": None, "rrule": None, "confidence": 0.7
            })
        return events

    for i, m in enumerate(tokens):
        mon_txt = m.group("mon").upper()
        if mon_txt not in MON:
            continue
        mon = MON[mon_txt]
        d1 = int(m.group("d1"))
        d2 = m.group("d2")

        seg_start = m.end()
        seg_end = tokens[i + 1].start() if i + 1 < len(tokens) else len(full)
        raw_title = full[seg_start:seg_end].strip()
        raw_title = re.sub(r"^[\W_0-9:\-\|/\\]+", "", raw_title)
        raw_title = raw_title.splitlines()[0] if raw_title else ""
        title = canon_title(raw_title)

        year = year_for_month(mon)
        start = datetime(year, mon, d1, 0, 0, 0, tzinfo=ZoneInfo(tz))
        if d2:
            end_day = int(re.sub(r"\D", "", d2)) if re.search(r"\D", d2 or "") else int(d2)
            end = datetime(year, mon, end_day, 0, 0, 0, tzinfo=ZoneInfo(tz)) + timedelta(days=1)
        else:
            end = start + timedelta(days=1)

        events.append({
            "title": title or "School Event",
            "start": start,
            "end": end,
            "all_day": True,
            "location": default_location,
            "notes": None,
            "rrule": None,
            "confidence": 0.9,
        })

    return events

# ---- ICS builder (robust) ----
def build_ics(approved: list[EventDraft]) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//Cal MVP//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    for e in approved:
        if not e.title or not e.start or not e.end:
            continue

        try:
            ve = Event()
            # unique + import-friendly
            ve.add("uid", f"{e.uid}-{e.id}@cal-mvp")
            ve.add("summary", e.title)
            ve.add("dtstamp", now)

            if e.all_day:
                ve.add("dtstart", e.start.date())
                ve.add("dtend", e.end.date())
            else:
                s = e.start
                t = e.end if e.end and e.end >= e.start else e.start
                ve.add("dtstart", s)
                ve.add("dtend", t)

            if e.location:
                ve.add("location", e.location)
            if e.notes:
                ve.add("description", e.notes)
            if e.rrule:
                ve.add("rrule", e.rrule)

            cal.add_component(ve)
        except Exception as ex:
            print("ICS build: skipped event due to error:", e.id, e.title, ex)

    return cal.to_ical()

# ---- API ----
app = FastAPI(title="Calendar Ingest MVP")

@app.post("/sources/upload")
async def upload_source(
    name: str = Form("Source"),
    tz: str = Form("America/New_York"),
    default_location: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    path = os.path.join(UPLOAD_DIR, token_urlsafe(8) + "_" + file.filename)
    with open(path, "wb") as f:
        f.write(content)

    src = Source(
        name=name,
        tz=tz,
        default_location=default_location,
        file_path=path,
        last_hash=sha256_hex(content),
        feed_token=token_urlsafe(24),
    )
    db.add(src)
    db.commit()
    db.refresh(src)

    is_pdf = file.filename.lower().endswith(".pdf")
    text = extract_text(pdf_bytes=content if is_pdf else None, img_bytes=content if not is_pdf else None)
    events = parse_events_from_text(text, tz, default_location)

    for ev in events:
        uid = stable_uid(ev["title"], ev["start"], ev["end"], ev["location"])
        db.add(EventDraft(source_id=src.id, uid=uid, approved=False, **ev))
    db.commit()

    return {
        "source_id": src.id,
        "events_created": len(events),
        "feed_url": f"/feeds/{src.feed_token}.ics",
    }

@app.get("/sources/{source_id}/events")
def list_events(source_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(EventDraft).where(EventDraft.source_id == source_id)).all()
    return [
        dict(
            id=r.id,
            title=r.title,
            start=r.start,
            end=r.end,
            all_day=r.all_day,
            location=r.location,
            notes=r.notes,
            rrule=r.rrule,
            confidence=r.confidence,
            approved=r.approved,
            uid=r.uid,
        )
        for r in rows
    ]

@app.post("/events/{event_id}/approve")
def approve_event(event_id: int, db: Session = Depends(get_db)):
    row = db.get(EventDraft, event_id)
    if not row:
        raise HTTPException(404, "Not found")
    row.approved = True
    db.commit()
    return {"ok": True}

@app.post("/sources/{source_id}/approve_all")
def approve_all(source_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(EventDraft).where(EventDraft.source_id == source_id)).all()
    for r in rows:
        r.approved = True
    db.commit()
    return {"approved_count": len(rows)}

@app.get("/feeds/{feed_token}.ics")
def serve_feed(feed_token: str, db: Session = Depends(get_db)):
    src = db.scalar(select(Source).where(Source.feed_token == feed_token))
    if not src:
        raise HTTPException(404, "Unknown feed")
    approved = db.scalars(
        select(EventDraft).where(EventDraft.source_id == src.id, EventDraft.approved == True)
    ).all()

    try:
        ics = build_ics(approved)
    except Exception as ex:
        print("ERROR building ICS:", ex)
        raise HTTPException(500, "Failed to build ICS")

    return Response(
        content=ics,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="calendar.ics"'},
    )

# Debug helper: see what will be exported
@app.get("/feeds/debug/{feed_token}")
def debug_feed(feed_token: str, db: Session = Depends(get_db)):
    src = db.scalar(select(Source).where(Source.feed_token == feed_token))
    if not src:
        raise HTTPException(404, "Unknown feed")
    rows = db.scalars(
        select(EventDraft).where(EventDraft.source_id == src.id, EventDraft.approved == True)
    ).all()
    return {
        "approved_count": len(rows),
        "sample": [
            dict(
                id=r.id,
                title=r.title,
                start=r.start.isoformat() if r.start else None,
                end=r.end.isoformat() if r.end else None,
                all_day=r.all_day,
            )
            for r in rows[:5]
        ],
    }
