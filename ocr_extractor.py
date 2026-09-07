"""
ocr_extractor.py
------------------
Extracts clinical lab values from an uploaded liver-function-test report
(image or PDF) using Tesseract OCR plus regex pattern matching.

This is best-effort extraction: lab report layouts vary wildly between
hospitals, so the parser looks for common label aliases and pulls the nearest
plausible number after each one. Fields it can't read confidently are left
out and the form falls back to manual entry for those.

Physical-exam and staging fields (Ascites, Hepatomegaly, Spiders, Edema,
Stage) never appear on a lab report, so those always need clinician input.
"""

import io
import os
import re
import shutil

from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:  # pragma: no cover
    PDF_SUPPORT = False


class OCRUnavailableError(RuntimeError):
    """Raised when the Tesseract binary can't be found.

    Kept distinct from a parse failure so the UI can tell the user to install
    Tesseract rather than blaming their scan quality.
    """


# Label aliases as they realistically appear on reports, checked in order.
FIELD_PATTERNS = {
    "age": [r"\bage\b"],
    "bilirubin": [
        r"total\s*bilirubin", r"bilirubin\s*[\(\-]?\s*total", r"serum\s*bilirubin",
        r"\bbilirubin\b", r"\bt\.?\s*bil\b", r"\bs\.?\s*bilirubin\b",
    ],
    "cholesterol": [r"total\s*cholesterol", r"serum\s*cholesterol", r"\bcholesterol\b", r"\bchol\b"],
    "albumin": [r"serum\s*albumin", r"\balbumin\b", r"\balb\b"],
    "copper": [r"urine\s*copper", r"\bcopper\b", r"\bcu\b"],
    "alk_phos": [
        r"alkaline\s*phosphatase", r"alk(?:aline)?\.?\s*phos(?:phatase)?",
        r"\balp\b", r"\bs\.?\s*alp\b",
    ],
    "sgot": [r"\bsgot\b", r"\bast\b", r"aspartate\s*(?:amino)?transferase", r"ast\s*\(?sgot\)?"],
    "triglycerides": [r"triglycerides?", r"\btgl?\b"],
    "platelets": [r"platelet\s*count", r"platelets?", r"\bplt\b"],
    "prothrombin": [r"prothrombin\s*time", r"\bpt\s*\(?inr\)?", r"\bprothrombin\b", r"\bpt\b(?!\w)"],
}

GENDER_PATTERNS = [
    (r"\b(?:female|f)\b", "F"),
    (r"\b(?:male|m)\b", "M"),
]

# Allow a label/value separator, and tolerate the OCR noise characters that
# commonly appear between a label and its number.
NUMBER_RE = r"[:\-=\s\.]{0,6}([0-9]{1,6}(?:[.,][0-9]{1,3})?)"

# Physiologically plausible ranges, used to sanity-check each read. Scans
# sometimes lose a decimal point ("3.4" -> "34"); if the raw value is out of
# range but a factor of ten brings it back in, assume that's what happened.
PLAUSIBLE_RANGES = {
    "age": (1, 120),
    "bilirubin": (0.1, 40),
    "cholesterol": (50, 1500),
    "albumin": (1.0, 6.5),
    "copper": (4, 650),
    "alk_phos": (30, 20000),
    "sgot": (5, 900),
    "triglycerides": (20, 1200),
    "platelets": (10, 1000),
    "prothrombin": (8, 35),
}

# Units near a number confirm the field, which disambiguates reports where the
# same number appears in a reference-range column.
UNIT_HINTS = {
    "bilirubin": ["mg/dl", "mg/dL", "umol/l"],
    "cholesterol": ["mg/dl", "mmol/l"],
    "albumin": ["g/dl", "gm/dl", "g/l"],
    "alk_phos": ["u/l", "iu/l"],
    "sgot": ["u/l", "iu/l", "u/ml"],
    "triglycerides": ["mg/dl"],
    "platelets": ["/ul", "lakh", "10^3", "10*3", "cumm", "/cmm"],
    "prothrombin": ["sec", "seconds", "s"],
}


def configure_tesseract(explicit_path: str = "") -> bool:
    """Point pytesseract at the Tesseract binary.

    Checks, in order: an explicit configured path, the system PATH, then the
    default Windows install locations (where the installer does not add itself
    to PATH). Returns True if a usable binary was found.
    """
    if not PYTESSERACT_AVAILABLE:
        return False

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    found_on_path = shutil.which("tesseract")
    if found_on_path:
        candidates.append(found_on_path)

    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True

    return False


def is_ocr_available() -> bool:
    """True when OCR can actually run right now."""
    if not PYTESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _sanity_correct(field: str, value: float):
    """Clamp a read into its plausible range, correcting decimal-point slips."""
    bounds = PLAUSIBLE_RANGES.get(field)
    if bounds is None:
        return value
    lo, hi = bounds
    if lo <= value <= hi:
        return value
    for factor in (1000, 100, 10, 0.1):
        adjusted = value / factor if factor >= 1 else value * 10
        if lo <= adjusted <= hi:
            return round(adjusted, 2)
    return None  # implausible even after correction; drop it


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Grayscale, upscale, autocontrast and sharpen — measurably better reads
    on phone photos of printed reports than feeding the raw image in."""
    image = image.convert("L")
    width, height = image.size
    if max(width, height) < 1800:
        scale = 1800 / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    image = ImageOps.autocontrast(image)
    return image.filter(ImageFilter.SHARPEN)


def _run_ocr(image: Image.Image) -> str:
    if not is_ocr_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed or not on PATH. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki (Windows) or via your "
            "package manager, then set TESSERACT_CMD if it isn't on PATH. "
            "Manual entry works without it."
        )
    processed = _preprocess_for_ocr(image)
    # PSM 6 assumes a uniform block of text, which suits tabular lab reports.
    return pytesseract.image_to_string(processed, config="--psm 6")


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """OCR an uploaded image, or the first two pages of a PDF."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        if not PDF_SUPPORT:
            raise OCRUnavailableError(
                "PDF support needs the pdf2image package and the Poppler binaries. "
                "Upload a JPG/PNG of the report instead, or enter values manually."
            )
        try:
            pages = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=2)
        except Exception as e:
            raise OCRUnavailableError(
                f"Could not read the PDF ({e}). Poppler may not be installed. "
                "Try uploading an image of the report instead."
            ) from e
        return "\n".join(_run_ocr(page) for page in pages)

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Exception as e:
        raise ValueError(f"That file isn't a readable image ({e}).") from e

    return _run_ocr(image)


def parse_lab_values(text: str) -> dict:
    """Pull whichever known lab fields the OCR text contains."""
    results: dict[str, float | str] = {}
    lower_text = text.lower().replace("|", " ")

    for field, aliases in FIELD_PATTERNS.items():
        for alias in aliases:
            for match in re.finditer(alias + r"[^0-9\n\r]{0,30}" + NUMBER_RE, lower_text):
                raw = match.group(1).replace(",", ".")
                try:
                    value = float(raw)
                except ValueError:
                    continue

                corrected = _sanity_correct(field, value)
                if corrected is None:
                    continue

                # Prefer a candidate whose expected unit appears just after it.
                tail = lower_text[match.end(): match.end() + 20]
                hints = UNIT_HINTS.get(field, [])
                confident = any(hint in tail for hint in hints) if hints else True

                if field not in results or confident:
                    results[field] = corrected
                if confident:
                    break
            if field in results:
                break

    for pattern, label in GENDER_PATTERNS:
        if re.search(r"(?:sex|gender)\s*[:\-]?\s*" + pattern, lower_text):
            results["gender"] = label
            break
        # Handle "Age / Gender : 52 Years / Male" format
        if re.search(r"(?:sex|gender)\s*[:\-]?\s*\d.*?(?:/|,)\s*" + pattern, lower_text):
            results["gender"] = label
            break

    # Extract patient name (single line only)
    for line in text.splitlines():
        name_match = re.search(
            r"(?:patient\s*name|name\s*of\s*patient|patient)\s*[:\-]?\s*"
            r"([A-Z][a-zA-Z .]+(?:\s+[A-Z][a-zA-Z .]+)*)",
            line,
            re.IGNORECASE,
        )
        if name_match:
            name = name_match.group(1).strip()
            if len(name) > 1 and name.lower() not in ("name", "id", "age"):
                results["patient_name"] = name
                break

    # Extract age from demographics line (e.g. "Age / Gender : 52 Years / Male")
    age_demo = re.search(
        r"(?:age\s*(?:/|&)\s*(?:gender|sex)|age)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?|y)?",
        lower_text,
    )
    if age_demo and "age" not in results:
        age_val = int(age_demo.group(1))
        if 1 <= age_val <= 120:
            results["age"] = age_val

    return results


def extract_lab_report(file_bytes: bytes, filename: str) -> dict:
    """Full pipeline: OCR the upload, parse it, and report what was found."""
    text = extract_text_from_upload(file_bytes, filename)
    values = parse_lab_values(text)
    return {
        "extracted_fields": values,
        "fields_found": len(values),
        "raw_text_preview": text.strip()[:800],
    }
