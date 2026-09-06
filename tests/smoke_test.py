"""
Smoke tests for the field-extraction logic.

These run WITHOUT PaddleOCR installed and without any real ID images: each
test feeds a hand-built list of bounding-box "items" that reconstructs a
real card layout, including the OCR noise actually observed in practice
("VALDTO" for "VALID TO", "eset" for "Address:", "Natlonality" for
"Nationality").

That matters because the geometry is the thing under test: a card is a 2-D
layout, and most extraction bugs are about which box belongs to which
label, not about the recogniser.

    python tests/smoke_test.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub PaddleOCR so the processors import without the heavy dependency.
if "paddleocr" not in sys.modules:
    _stub = types.ModuleType("paddleocr")

    class _StubPaddleOCR:
        def __init__(self, *args, **kwargs):
            pass

    _stub.PaddleOCR = _StubPaddleOCR
    sys.modules["paddleocr"] = _stub

from id_extraction import ocr_processors as ocr  # noqa: E402
from id_extraction import service  # noqa: E402


def items(spec):
    """(text, x1, y1, x2, y2) -> the item dicts the processors consume."""
    return [
        {
            "text": text, "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "xc": (x1 + x2) / 2.0, "yc": (y1 + y2) / 2.0,
        }
        for text, x1, y1, x2, y2 in spec
    ]


def check(name, actual, expected):
    failures = [
        f"      {k}: got {actual.get(k)!r}, want {v!r}"
        for k, v in expected.items()
        if actual.get(k) != v
    ]
    if failures:
        print(f"  FAIL  {name}")
        print("\n".join(failures))
        return False
    print(f"  ok    {name}")
    return True


# ---------------------------------------------------------------------------
# Fixtures: real card layouts, with real observed OCR noise
# ---------------------------------------------------------------------------

MEDICARE = items([
    ("medicare", 330, 20, 470, 60),
    ("1234", 150, 90, 250, 125), ("56789", 265, 90, 390, 125), ("0", 405, 90, 430, 125),
    ("1", 60, 140, 75, 162), ("JOHN", 95, 140, 175, 162), ("SMITH", 230, 140, 320, 162),
    ("2", 60, 168, 75, 190), ("HELEN", 95, 168, 190, 190), ("SMITH", 230, 168, 320, 190),
    ("4", 60, 224, 75, 246), ("JESSICA", 95, 224, 210, 246), ("SMITH", 230, 224, 320, 246),
    ("VALDTO", 300, 290, 390, 312),   # OCR garbled "VALID TO"
    ("11/10", 400, 290, 470, 312),
])

LICENCE_INLINE = items([   # federal-style specimen, "label: value" in one box
    ("ec", 95, 5, 110, 14),                       # OCR noise
    ("Australia", 285, 30, 330, 42),
    ("Mrs.Grey N.Nomad", 18, 68, 130, 82),
    ("Licence No", 283, 66, 340, 76), ("00112233AU", 277, 78, 345, 90),
    ("eset", 18, 90, 45, 100),                    # OCR garbled "Address:"
    ("c/o The Open Road", 18, 100, 100, 112), ("Australia", 18, 112, 60, 124),
    ("Birthdate:24/3/1937", 18, 130, 120, 142), ("Conditions:S", 140, 130, 205, 142),
    ("Licence Class:C", 18, 148, 105, 160), ("Donor:A", 140, 148, 185, 160),
    ("Valid in:", 18, 166, 60, 178), ("Expires", 140, 166, 180, 178),
    ("ACT,NSW,NT,QLD", 18, 180, 110, 192), ("11/1/0", 140, 180, 175, 192),
    ("SA,TAS,VIC.WA", 18, 194, 100, 206),
])

LICENCE_2D = items([   # NSW-style: labels and values stacked in two columns
    ("Driver Licence", 250, 10, 420, 35), ("New South Wales, Australia", 245, 36, 430, 55),
    ("Given Name", 25, 60, 115, 75), ("Family Name", 25, 78, 115, 93),   # annotation callouts
    ("Card Number", 455, 60, 540, 75), ("9 999 999 999", 450, 76, 545, 96),
    ("CENTENNIAL PLAZA", 25, 105, 190, 125), ("260 ELIZABETH ST", 25, 127, 185, 147),
    ("SURRY HILLS 2010 NSW", 25, 149, 215, 169),
    ("Licence No.", 25, 172, 105, 187), ("Donor", 185, 172, 230, 187),
    ("Licence Number", 28, 190, 125, 205), ("A", 188, 188, 200, 208),
    ("Licence Class", 25, 208, 110, 223), ("Conditions", 185, 208, 255, 223),
    ("C", 28, 226, 40, 246), ("X", 188, 226, 200, 246),
    ("Richard Plantagenet Campbell", 345, 255, 500, 272),
    ("Expiry Date", 420, 275, 490, 290), ("01 JAN 2000", 415, 292, 500, 312),
])

PASSPORT = items([   # bilingual labels, values stacked below
    ("Passport No./No du passeport", 150, 40, 330, 56), ("PA1234567", 150, 58, 250, 74),
    ("Surname/Nom", 20, 90, 110, 106), ("CAMPBELL", 20, 108, 120, 124),
    ("Given Names/Prenoms", 20, 140, 160, 156), ("RICHARD PLANTAGENET", 20, 158, 220, 174),
    ("Natlonality/Nationalite", 20, 190, 150, 206), ("AUSTRALIAN", 20, 208, 130, 224),  # garbled
    ("Place of birth/Lieu de naissance", 20, 290, 230, 306), ("SYDNEY", 20, 308, 90, 324),
    ("Date of lssue/Date de delivrance", 20, 340, 230, 356), ("01 JAN 2020", 20, 358, 120, 374),  # garbled
    ("Date of expiry/Date d'expiration", 20, 390, 230, 406), ("01 JAN 2030", 20, 408, 120, 424),
])

PASSPORT_MRZ = PASSPORT + items([
    ("P<AUSCAMPBELL<<RICHARD<PLANTAGENET<<<<<<<<<<<", 20, 460, 500, 478),
    ("PA12345674AUS9001013M3001017<<<<<<<<<<<<<<02", 20, 480, 500, 498),
])


def main():
    results = []
    new = object.__new__

    print("\nExtraction")
    results.append(check(
        "medicare (garbled 'VALDTO' label)",
        new(ocr.MedicareProcessor)._process_ocr_items(MEDICARE),
        {"Medicare Number": "1234 56789 0", "Valid To": "11/10",
         "Cardholder 1": "JOHN SMITH", "Cardholder 4": "JESSICA SMITH"},
    ))
    results.append(check(
        "licence, inline labels (garbled 'eset' address label)",
        new(ocr.DrivingLicenceProcessor)._process_ocr_items(LICENCE_INLINE),
        {"Name": "Mrs.Grey N.Nomad", "Address": "c/o The Open Road, Australia",
         "Licence Number": "00112233AU", "Licence Class": "C", "Conditions": "S",
         "Donor": "A", "Date of Birth": "24/3/1937"},
    ))
    results.append(check(
        "licence, 2-D layout with annotation callouts",
        new(ocr.DrivingLicenceProcessor)._process_ocr_items(LICENCE_2D),
        {"Licence Class": "C", "Conditions": "X", "Donor": "A",
         "Card Number": "9 999 999 999", "Expiry Date": "01 JAN 2000",
         "Name": "Richard Plantagenet Campbell",
         "Address": "CENTENNIAL PLAZA, 260 ELIZABETH ST, SURRY HILLS 2010 NSW"},
    ))
    results.append(check(
        "passport, no MRZ, garbled bilingual labels",
        new(ocr.PassportProcessor)._process_ocr_items(PASSPORT),
        {"Surname": "CAMPBELL", "Given Names": "RICHARD PLANTAGENET",
         "Passport Number": "PA1234567", "Nationality": "AUSTRALIAN",
         "Place of Birth": "SYDNEY", "Date of Issue": "01 JAN 2020",
         "Date of Expiry": "01 JAN 2030"},
    ))
    results.append(check(
        "passport, MRZ takes precedence (expiry is 2030, not 1930)",
        new(ocr.PassportProcessor)._process_ocr_items(PASSPORT_MRZ),
        {"Surname": "CAMPBELL", "Passport Number": "PA1234567",
         "Date of Birth": "01/01/1990", "Date of Expiry": "01/01/2030", "Sex": "M"},
    ))

    print("\nSubmission validation")
    cases = [
        ("medicare accepts MM/YY", "medicare",
         {"Medicare Number": "1234 56789 0", "Valid To": "11/10"}, {}),
        ("medicare rejects a malformed number", "medicare",
         {"Medicare Number": "12345"},
         {"Medicare Number": "Expected 10 digits, e.g. 1234 56789 0."}),
        ("licence accepts '01 JAN 2000'", "licence",
         {"Licence Number": "123", "Name": "A B", "Expiry Date": "01 JAN 2000"}, {}),
        ("licence accepts DD/MM/YYYY", "licence",
         {"Licence Number": "123", "Name": "A B", "Expiry Date": "9/1/13"}, {}),
        ("passport flags missing required fields", "passport",
         {"Surname": "CAMPBELL"},
         {"Passport Number": "This field is required.",
          "Given Names": "This field is required.",
          "Date of Expiry": "This field is required."}),
    ]
    for name, doc_type, fields, expected in cases:
        actual = service.validate_submission(doc_type, fields)
        ok = actual == expected
        print(f"  {'ok   ' if ok else 'FAIL '} {name}")
        if not ok:
            print(f"      got {actual}, want {expected}")
        results.append(ok)

    print("\n" + "=" * 62)
    passed = sum(1 for r in results if r)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
