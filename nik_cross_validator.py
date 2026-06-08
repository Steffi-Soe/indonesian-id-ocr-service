"""
nik_cross_validator.py
----------------------
Bidirectional NIK ↔ KTP field validation and repair.

NIK structure (16 digits):
  [PP][KK][DD][OB][MM][YY][SSSS]

  PP   1–2    Province code
  KK   3–4    City/Regency code
  DD   5–6    District code
  OB   7–8    Day of birth  (male 01-31 | female 41-71 = day + 40)
  MM   9–10   Month (01-12)
  YY   11–12  Last two digits of birth year
  SSSS 13–16  Sequence number

Two repair directions:
  NIK → Fields  (when NIK is 16 clean digits, use it as ground truth)
  Fields → NIK  (when NIK is malformed, use DOB / gender to reconstruct)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

_MIN_BIRTH_YEAR = 1920
_REFERENCE_YEAR = 2026


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CrossValResult:
    nik_corrected:    bool       = False
    gender_corrected: bool       = False
    date_corrected:   bool       = False
    conflicts:        List[str]  = field(default_factory=list)
    confirmations:    List[str]  = field(default_factory=list)
    confidence_delta: float      = 0.0   # net bonus (+) / penalty (-)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class NIKCrossValidator:
    """
    Validates and repairs a raw KTP extraction dict using bidirectional
    NIK ↔ demographic-field logic.

    Usage
    -----
        validator = NIKCrossValidator()
        repaired  = validator.validate_and_repair(raw_ktp_dict)
        # repaired["_cross_val"] → CrossValResult (popped by caller)
    """

    def validate_and_repair(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return data

        repaired = dict(data)
        nik      = str(repaired.get("NIK") or "")

        if re.match(r"^\d{16}$", nik):
            repaired, result = self._nik_to_fields(repaired, nik)
        else:
            repaired, result = self._fields_to_nik(repaired, nik)

        repaired["_cross_val"] = result
        return repaired

    # ------------------------------------------------------------------
    # Direction 1 – NIK is authoritative
    # ------------------------------------------------------------------

    def _nik_to_fields(
        self, data: Dict[str, Any], nik: str
    ) -> Tuple[Dict[str, Any], CrossValResult]:
        result = CrossValResult()

        day_raw = int(nik[6:8])
        month   = int(nik[8:10])
        year_2  = int(nik[10:12])

        is_female  = day_raw > 40
        day        = day_raw - 40 if is_female else day_raw
        gender_nik = "PEREMPUAN" if is_female else "LAKI-LAKI"

        # Reconstruct full birth year
        current_yy = _REFERENCE_YEAR % 100
        year = (2000 + year_2) if year_2 <= current_yy else (1900 + year_2)

        nik_date     = f"{day:02d}-{month:02d}-{year:04d}"
        date_valid   = (
            1 <= day <= 31
            and 1 <= month <= 12
            and _MIN_BIRTH_YEAR <= year <= _REFERENCE_YEAR
        )

        # ---- Province code plausibility ----
        prov = int(nik[0:2])
        if 11 <= prov <= 94:
            result.confidence_delta += 0.03
        else:
            result.conflicts.append(f"Province code {prov} outside 11–94")
            result.confidence_delta -= 0.05

        # ---- Sequence non-zero ----
        seq = int(nik[12:16])
        if seq == 0:
            result.conflicts.append("Sequence 0000 is unusual")
            result.confidence_delta -= 0.02

        # ---- Validate day/month encoding ----
        if not date_valid:
            result.conflicts.append(
                f"NIK encodes implausible date: day={day} month={month} year={year}"
            )
            result.confidence_delta -= 0.10
            return data, result

        result.confidence_delta += 0.03   # NIK date is internally plausible

        # ---- Gender cross-check ----
        existing_gender = self._normalise_gender(data.get("Jenis Kelamin"))
        if not existing_gender:
            data["Jenis Kelamin"] = gender_nik
            result.confirmations.append(f"Gender inferred from NIK: {gender_nik}")
            result.confidence_delta += 0.04
        elif existing_gender == gender_nik:
            result.confirmations.append(f"Gender matches NIK encoding: {gender_nik}")
            result.confidence_delta += 0.05
        else:
            result.conflicts.append(
                f"Gender mismatch — OCR: '{existing_gender}'  NIK: '{gender_nik}'"
            )
            data["Jenis Kelamin"] = gender_nik   # NIK is authoritative
            result.gender_corrected = True
            result.confidence_delta -= 0.04

        # ---- Date cross-check ----
        existing_date = self._extract_date(data.get("Tempat/Tgl Lahir", ""))

        if not existing_date:
            # No OCR date available — inject NIK-derived value
            place = self._extract_place(data.get("Tempat/Tgl Lahir", ""))
            ttl_new = f"{place},{nik_date}" if place else nik_date
            data["Tempat/Tgl Lahir"] = ttl_new
            result.date_corrected = True
            result.confirmations.append(f"Date injected from NIK: {nik_date}")
            result.confidence_delta += 0.04
        elif existing_date == nik_date:
            result.confirmations.append(f"Date fully matches NIK: {nik_date}")
            result.confidence_delta += 0.08
        else:
            e_day, e_mon, e_yr = self._parse_dmy(existing_date)
            match_yr  = (e_yr  is not None) and ((e_yr  % 100) == year_2)
            match_mon = (e_mon is not None) and (e_mon == month)
            match_day = (e_day is not None) and (e_day == day)

            if match_yr and match_mon and match_day:
                # All components match; only formatting differed
                result.confirmations.append(f"Date matches NIK (format corrected): {nik_date}")
                result.confidence_delta += 0.06
                place = self._extract_place(data.get("Tempat/Tgl Lahir", ""))
                ttl_new = f"{place},{nik_date}" if place else nik_date
                data["Tempat/Tgl Lahir"] = ttl_new
                result.date_corrected = True
            elif match_yr:
                result.confirmations.append(f"Date year-suffix matches NIK ({year_2:02d})")
                result.confidence_delta += 0.03
            else:
                result.conflicts.append(
                    f"Date mismatch — OCR: '{existing_date}'  NIK: '{nik_date}'"
                )
                result.confidence_delta -= 0.05
                # Correct date using NIK (NIK is ground truth)
                place = self._extract_place(data.get("Tempat/Tgl Lahir", ""))
                ttl_new = f"{place},{nik_date}" if place else nik_date
                data["Tempat/Tgl Lahir"] = ttl_new
                result.date_corrected = True
                logger.info(
                    "Date corrected by NIK: '%s' → '%s'", existing_date, nik_date
                )

        return data, result

    # ------------------------------------------------------------------
    # Direction 2 – extracted fields drive NIK repair
    # ------------------------------------------------------------------

    def _fields_to_nik(
        self, data: Dict[str, Any], raw_nik: str
    ) -> Tuple[Dict[str, Any], CrossValResult]:
        result = CrossValResult()

        existing_date = self._extract_date(data.get("Tempat/Tgl Lahir", ""))
        if not existing_date:
            result.conflicts.append("No usable date found; cannot repair NIK")
            return data, result

        e_day, e_mon, e_yr = self._parse_dmy(existing_date)
        if e_day is None:
            return data, result

        gender_norm = self._normalise_gender(data.get("Jenis Kelamin"))
        is_female   = gender_norm == "PEREMPUAN"
        enc_day     = (e_day + 40) if is_female else e_day
        e_yr2       = e_yr % 100
        expected_dob = f"{enc_day:02d}{e_mon:02d}{e_yr2:02d}"

        if not raw_nik:
            result.conflicts.append("NIK missing; cannot attempt field-driven repair")
            return data, result

        # Apply OCR character substitution to raw NIK string
        from nik_fuzzy import OCR_TO_DIGIT
        substituted = "".join(OCR_TO_DIGIT.get(c, c) for c in raw_nik)
        digits      = re.sub(r"\D", "", substituted)

        if len(digits) == 16:
            actual_dob = digits[6:12]
            if actual_dob == expected_dob:
                result.confirmations.append(
                    "NIK DOB segment matches extracted date ✓"
                )
                result.confidence_delta += 0.08
                if digits != raw_nik:
                    data["NIK"] = digits
                    result.nik_corrected = True
            else:
                result.conflicts.append(
                    f"NIK DOB '{actual_dob}' ≠ expected '{expected_dob}'"
                )
                result.confidence_delta -= 0.05

        elif len(digits) == 15:
            repaired = self._pad_nik_with_dob(digits, expected_dob)
            if repaired:
                data["NIK"] = repaired
                result.nik_corrected = True
                result.confidence_delta += 0.04
                result.confirmations.append(
                    f"NIK reconstructed from DOB: '{raw_nik}' → '{repaired}'"
                )
                logger.info("NIK repaired (15→16): '%s' → '%s'", raw_nik, repaired)
            else:
                result.conflicts.append(
                    f"15-digit NIK '{digits}' could not be padded to match DOB '{expected_dob}'"
                )
                result.confidence_delta -= 0.05

        else:
            result.conflicts.append(
                f"NIK has {len(digits)} digits after substitution; cannot validate"
            )
            result.confidence_delta -= 0.08

        return data, result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_nik_with_dob(digits_15: str, expected_dob: str) -> Optional[str]:
        """
        Insert one digit at each of the 16 possible positions and return
        the first 16-digit candidate whose DOB segment (chars 6–11) matches.
        Tries leading positions first since a dropped leading digit is the
        most common OCR truncation.
        """
        priority = list(range(16))
        for digit in "0123456789":
            for pos in priority:
                candidate = digits_15[:pos] + digit + digits_15[pos:]
                if len(candidate) == 16 and candidate[6:12] == expected_dob:
                    return candidate
        return None

    @staticmethod
    def _extract_date(ttl: str) -> Optional[str]:
        """Return the first DD-MM-YYYY token found in ttl, or None."""
        if not ttl:
            return None
        m = re.search(r"\b(\d{2})-(\d{2})-(\d{4})\b", str(ttl))
        return m.group(0) if m else None

    @staticmethod
    def _extract_place(ttl: str) -> Optional[str]:
        """Return the text that precedes the date token, stripped of punctuation."""
        if not ttl:
            return None
        m = re.search(r"\b\d{2}-\d{2}-\d{4}\b", str(ttl))
        if not m:
            return None
        place = str(ttl)[: m.start()].strip().strip(",.:- ")
        return place if len(place) >= 2 else None

    @staticmethod
    def _parse_dmy(
        date_str: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", date_str)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        return None, None, None

    @staticmethod
    def _normalise_gender(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        v = str(raw).upper().strip()
        if any(k in v for k in ("LAKI", "PRIA", "MALE", "LK")):
            return "LAKI-LAKI"
        if any(k in v for k in ("PEREMPUAN", "WANITA", "FEMALE", "PR")):
            return "PEREMPUAN"
        return None