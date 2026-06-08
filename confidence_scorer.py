"""
confidence_scorer.py
--------------------
Unified confidence scoring and cross-field validation for KTP OCR output.

Provides:
  * Per-field confidence scores (NIK, Nama, date, address, etc.)
  * Cross-field consistency checks (NIK birth-date ↔ Tgl Lahir field)
  * NIK structural bonus/penalty tiers
  * Document-level composite score
  * Actionable quality report (which fields are missing / suspect)

Usage
-----
    scorer = KTPConfidenceScorer()
    report = scorer.score(extracted_data_dict)
    print(report.overall)              # 0.0 – 1.0
    print(report.missing_critical)     # list of critical missing fields
    print(report.cross_check_passed)   # True/False
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FieldScore:
    name:       str
    value:      Optional[str]
    score:      float          # 0.0 – 1.0
    issues:     List[str]      # human-readable problem descriptions


@dataclass
class DocumentReport:
    field_scores:        Dict[str, FieldScore]
    overall:             float          # weighted composite [0.0–1.0]
    field_count:         int
    max_field_count:     int
    missing_critical:    List[str]      # critical fields with score == 0
    low_confidence:      List[str]      # fields with score < 0.5
    cross_check_passed:  bool           # NIK ↔ birth-date consistency
    cross_check_notes:   List[str]
    grade:               str            # A/B/C/D/F
    # Breakdown of confidence adjustments from cross-validation
    nik_structural_bonus: float = 0.0
    cross_val_bonus:      float = 0.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class KTPConfidenceScorer:
    """
    Scores an extracted KTP data dictionary.

    The input dict uses the same key names as format_to_target_json():
        data = {
            "nomor":            "3201...",
            "nama":             "BUDI SANTOSO",
            "tempat_lahir":     "JAKARTA",
            "tgl_lahir":        "15-08-1990",
            "jenis_kelamin":    "LAKI-LAKI",
            ...
        }
    """

    # Weights for the composite score
    FIELD_WEIGHTS = {
        "nomor":             3.0,
        "nama":              2.5,
        "tgl_lahir":         1.5,
        "tempat_lahir":      0.8,
        "jenis_kelamin":     1.0,
        "agama":             0.5,
        "status_perkawinan": 0.5,
        "pekerjaan":         0.5,
        "kewarganegaraan":   0.5,
        "alamat.name":       1.0,
        "alamat.rt_rw":      0.5,
        "alamat.kel_desa":   0.7,
        "alamat.kecamatan":  0.7,
        "alamat.kabupaten":  0.7,
        "alamat.provinsi":   0.5,
    }

    CRITICAL_FIELDS = {"nomor", "nama", "tgl_lahir"}

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def score(self, data: Dict[str, Any]) -> DocumentReport:
        if not data:
            return self._empty_report()

        field_scores: Dict[str, FieldScore] = {}

        addr = data.get("alamat") or {}
        if not isinstance(addr, dict):
            addr = {}

        flat: Dict[str, Optional[str]] = {
            "nomor":             data.get("nomor"),
            "nama":              data.get("nama"),
            "tgl_lahir":         data.get("tgl_lahir"),
            "tempat_lahir":      data.get("tempat_lahir"),
            "jenis_kelamin":     data.get("jenis_kelamin"),
            "agama":             data.get("agama"),
            "status_perkawinan": data.get("status_perkawinan"),
            "pekerjaan":         data.get("pekerjaan"),
            "kewarganegaraan":   data.get("kewarganegaraan"),
            "alamat.name":       addr.get("name"),
            "alamat.rt_rw":      addr.get("rt_rw"),
            "alamat.kel_desa":   addr.get("kel_desa"),
            "alamat.kecamatan":  addr.get("kecamatan"),
            "alamat.kabupaten":  addr.get("kabupaten"),
            "alamat.provinsi":   addr.get("provinsi"),
        }

        scorers = {
            "nomor":             self._score_nik,
            "nama":              self._score_nama,
            "tgl_lahir":         self._score_date,
            "tempat_lahir":      self._score_place_name,
            "jenis_kelamin":     self._score_jenis_kelamin,
            "agama":             self._score_enum_field,
            "status_perkawinan": self._score_enum_field,
            "pekerjaan":         self._score_free_text,
            "kewarganegaraan":   self._score_enum_field,
            "alamat.name":       self._score_address_name,
            "alamat.rt_rw":      self._score_rt_rw,
            "alamat.kel_desa":   self._score_free_text,
            "alamat.kecamatan":  self._score_free_text,
            "alamat.kabupaten":  self._score_kabupaten,
            "alamat.provinsi":   self._score_free_text,
        }

        for fname, value in flat.items():
            fs = scorers.get(fname, self._score_free_text)(fname, value)
            field_scores[fname] = fs

        # Weighted composite
        total_weight = sum(self.FIELD_WEIGHTS.values())
        weighted_sum = sum(
            field_scores[f].score * self.FIELD_WEIGHTS[f]
            for f in self.FIELD_WEIGHTS
        )
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # NIK structural bonus (separate from per-field score)
        nik_val = flat.get("nomor") or ""
        nik_structural_bonus = self._compute_nik_structural_bonus(nik_val)
        overall = min(1.0, overall + nik_structural_bonus)

        missing_critical = [
            f for f in self.CRITICAL_FIELDS
            if field_scores.get(f, FieldScore(f, None, 0.0, [])).score == 0.0
        ]
        low_confidence = [
            f for f in self.FIELD_WEIGHTS
            if 0 < field_scores.get(f, FieldScore(f, None, 0.0, [])).score < 0.5
        ]
        field_count = sum(
            1 for f in self.FIELD_WEIGHTS
            if field_scores.get(f, FieldScore(f, None, 0.0, [])).score > 0.0
        )

        cross_ok, cross_notes = self._cross_check(flat)
        grade = self._grade(overall, missing_critical)

        return DocumentReport(
            field_scores=field_scores,
            overall=float(overall),
            field_count=field_count,
            max_field_count=len(self.FIELD_WEIGHTS),
            missing_critical=missing_critical,
            low_confidence=low_confidence,
            cross_check_passed=cross_ok,
            cross_check_notes=cross_notes,
            grade=grade,
            nik_structural_bonus=nik_structural_bonus,
        )

    # -----------------------------------------------------------------------
    # NIK structural bonus
    # -----------------------------------------------------------------------

    def _compute_nik_structural_bonus(self, nik: str) -> float:
        """
        Additional document-level bonus / penalty based on NIK structural
        checks that go beyond whether the field itself is present.

        Bonuses:
          +0.03  province code in valid 11–94 range
          +0.02  day-of-birth encoding valid (01-31 or 41-71)
          +0.02  month encoding valid (01-12)
          +0.01  sequence non-zero

        Penalties:
          -0.05  invalid province code
          -0.08  invalid day-of-birth encoding
          -0.08  invalid month encoding
        """
        if not nik or not re.match(r'^\d{16}$', str(nik)):
            return 0.0

        bonus = 0.0
        prov  = int(nik[0:2])
        day   = int(nik[6:8])
        month = int(nik[8:10])
        seq   = int(nik[12:16])

        # Province
        if 11 <= prov <= 94:
            bonus += 0.03
        else:
            bonus -= 0.05

        # Day of birth
        if (1 <= day <= 31) or (41 <= day <= 71):
            bonus += 0.02
        else:
            bonus -= 0.08

        # Month
        if 1 <= month <= 12:
            bonus += 0.02
        else:
            bonus -= 0.08

        # Sequence
        if seq > 0:
            bonus += 0.01

        return float(bonus)

    # -----------------------------------------------------------------------
    # Field scorers
    # -----------------------------------------------------------------------

    def _score_nik(self, name: str, value: Optional[str]) -> FieldScore:
        issues = []
        if not value:
            return FieldScore(name, value, 0.0, ["NIK missing"])

        nik = str(value)
        if not re.match(r'^\d{16}$', nik):
            # Partial credit if it has 16 chars but some are not digits
            digit_count = sum(1 for c in nik if c.isdigit())
            partial = 0.1 + 0.1 * (digit_count / 16)
            return FieldScore(name, value, round(partial, 2), ["NIK not 16 digits"])

        score = 1.0
        prov  = int(nik[0:2])
        day   = int(nik[6:8])
        month = int(nik[8:10])
        seq   = int(nik[12:16])

        if prov < 11 or prov > 94:
            issues.append(f"Province code {prov} out of range 11–94")
            score *= 0.65

        if not ((1 <= day <= 31) or (41 <= day <= 71)):
            issues.append(f"Day-of-birth encoding {day} invalid (expected 01-31 or 41-71)")
            score = 0.0

        if month < 1 or month > 12:
            issues.append(f"Month encoding {month} invalid (expected 01-12)")
            score = 0.0

        if seq == 0:
            issues.append("Sequence 0000 is unusual")
            score *= 0.75

        return FieldScore(name, value, float(score), issues)

    def _score_nama(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Name missing"])
        v = str(value).strip()
        if len(v) < 2:
            return FieldScore(name, value, 0.1, ["Name too short"])
        issues = []
        alpha_ratio = sum(1 for c in v if c.isalpha() or c == ' ') / len(v)
        if alpha_ratio < 0.85:
            issues.append("Name contains unexpected characters")
        score = min(1.0, alpha_ratio)
        if re.search(r'\d', v):
            score *= 0.6
            issues.append("Name contains digits (OCR noise?)")
        return FieldScore(name, value, float(score), issues)

    def _score_date(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Date missing"])
        m = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', str(value))
        if not m:
            return FieldScore(
                name, value, 0.3,
                ["Date not in DD-MM-YYYY format; expected zero-padded DD-MM-YYYY"]
            )
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        issues = []
        score  = 1.0
        if not (1 <= day <= 31):
            issues.append(f"Day {day} out of range")
            score = 0.0
        if not (1 <= month <= 12):
            issues.append(f"Month {month} out of range")
            score = 0.0
        if not (1920 <= year <= 2010):
            if 2010 < year <= 2025:
                issues.append("Birth year implies very young person")
                score *= 0.7
            else:
                issues.append(f"Birth year {year} unrealistic")
                score *= 0.2
        return FieldScore(name, value, float(score), issues)

    def _score_place_name(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).strip()
        if len(v) < 3:
            return FieldScore(name, value, 0.2, ["Too short"])
        digit_ratio = sum(1 for c in v if c.isdigit()) / len(v)
        score = 1.0 - digit_ratio * 0.8
        issues = [] if digit_ratio == 0 else ["Contains digits (OCR noise?)"]
        return FieldScore(name, value, float(score), issues)

    def _score_jenis_kelamin(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).upper().strip()
        if v in ("LAKI-LAKI", "PEREMPUAN"):
            return FieldScore(name, value, 1.0, [])
        if "LAKI" in v or "MALE" in v or "PRIA" in v:
            return FieldScore(name, value, 0.8, ["Non-canonical form"])
        if "PEREMPUAN" in v or "FEMALE" in v or "WANITA" in v:
            return FieldScore(name, value, 0.8, ["Non-canonical form"])
        return FieldScore(name, value, 0.3, ["Unrecognised gender value"])

    def _score_enum_field(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).strip()
        if len(v) < 2:
            return FieldScore(name, value, 0.2, ["Too short"])
        return FieldScore(name, value, 0.9, [])

    def _score_free_text(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).strip()
        score = min(1.0, len(v) / 5)
        issues = []
        if re.match(r'^\d+$', v):
            issues.append("Value is all digits (likely OCR mis-assignment)")
            score *= 0.3
        return FieldScore(name, value, float(score), issues)

    def _score_address_name(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).strip()
        score = 1.0
        issues = []
        if len(v) < 5:
            score *= 0.4
            issues.append("Address name very short")
        return FieldScore(name, value, float(score), issues)

    def _score_rt_rw(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        m = re.match(r'^(\d{1,3})/(\d{1,3})$', str(value).strip())
        if not m:
            return FieldScore(name, value, 0.4, ["RT/RW not in NNN/NNN format"])
        return FieldScore(name, value, 1.0, [])

    def _score_kabupaten(self, name: str, value: Optional[str]) -> FieldScore:
        if not value:
            return FieldScore(name, value, 0.0, ["Missing"])
        v = str(value).strip().upper()
        if re.match(r'^[A-Z]{1,3}$', v):
            return FieldScore(name, value, 0.2, ["Kabupaten too short — OCR artefact"])
        if any(k in v for k in ("KOTA", "KAB", "KABUPATEN", "JAKARTA")):
            return FieldScore(name, value, 1.0, [])
        return FieldScore(name, value, 0.7, [])

    # -----------------------------------------------------------------------
    # Cross-field consistency — NIK ↔ Tgl Lahir ↔ Jenis Kelamin
    # -----------------------------------------------------------------------

    def _cross_check(self, flat: Dict[str, Optional[str]]):
        """
        Verify NIK birth-date and gender encoding match extracted fields.

        NIK digits 7-8  = day-of-birth (or day+40 for women)
        NIK digits 9-10 = month
        NIK digits 11-12 = last 2 digits of year
        Day > 40 → female
        """
        notes  = []
        passed = True

        nik = flat.get("nomor")
        tgl = flat.get("tgl_lahir")
        jk  = flat.get("jenis_kelamin", "") or ""

        if nik and tgl and re.match(r'^\d{16}$', str(nik)):
            m = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', str(tgl))
            if m:
                t_day = int(m.group(1))
                t_mon = int(m.group(2))
                t_yr  = int(m.group(3))
                n_day = int(nik[6:8])
                n_mon = int(nik[8:10])
                n_yr  = int(nik[10:12])
                t_yr2 = t_yr % 100

                # Determine gender encoding
                nik_is_female = n_day > 40
                adj_day       = n_day - 40 if nik_is_female else n_day

                # ---- Day check ----
                if adj_day != t_day:
                    notes.append(
                        f"NIK day-of-birth ({adj_day}) ≠ Tgl Lahir day ({t_day})"
                    )
                    passed = False

                # ---- Month check ----
                if n_mon != t_mon:
                    notes.append(
                        f"NIK month ({n_mon}) ≠ Tgl Lahir month ({t_mon})"
                    )
                    passed = False

                # ---- Year check ----
                if n_yr != t_yr2:
                    notes.append(
                        f"NIK year suffix ({n_yr:02d}) ≠ Tgl Lahir year suffix ({t_yr2:02d})"
                    )
                    passed = False

                # ---- Gender check ----
                jk_upper      = jk.upper()
                ocr_is_female = "PEREMPUAN" in jk_upper or "WANITA" in jk_upper
                if jk and (nik_is_female != ocr_is_female):
                    notes.append(
                        f"NIK gender encoding ({'female' if nik_is_female else 'male'}) "
                        f"≠ Jenis Kelamin ('{jk}')"
                    )
                    passed = False
                elif jk:
                    notes.append(
                        f"NIK gender encoding consistent with Jenis Kelamin ✓"
                    )

                if passed:
                    notes.append("NIK ↔ Tgl Lahir ↔ Jenis Kelamin all consistent ✓")

        elif not nik:
            notes.append("NIK missing; cross-check skipped")
        elif not tgl:
            notes.append("Tgl Lahir missing; cross-check skipped")

        return passed, notes

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _grade(overall: float, missing_critical: List[str]) -> str:
        if missing_critical:
            return "F" if len(missing_critical) >= 2 else "D"
        if overall >= 0.88: return "A"
        if overall >= 0.72: return "B"
        if overall >= 0.55: return "C"
        if overall >= 0.35: return "D"
        return "F"

    def _empty_report(self) -> DocumentReport:
        empty_scores = {
            f: FieldScore(f, None, 0.0, ["No data"])
            for f in self.FIELD_WEIGHTS
        }
        return DocumentReport(
            field_scores=empty_scores,
            overall=0.0,
            field_count=0,
            max_field_count=len(self.FIELD_WEIGHTS),
            missing_critical=list(self.CRITICAL_FIELDS),
            low_confidence=[],
            cross_check_passed=False,
            cross_check_notes=["No data to validate"],
            grade="F",
        )


# ---------------------------------------------------------------------------
# Quick summary printer
# ---------------------------------------------------------------------------

def print_report(report: DocumentReport) -> None:
    print(f"\n{'='*60}")
    print(f"  KTP Extraction Quality Report  —  Grade: {report.grade}")
    print(f"  Overall score      : {report.overall:.3f}")
    print(f"  NIK struct bonus   : {report.nik_structural_bonus:+.3f}")
    print(f"  Fields found       : {report.field_count}/{report.max_field_count}")
    print(f"  Cross-check        : {'PASS' if report.cross_check_passed else 'FAIL'}")
    for note in report.cross_check_notes:
        print(f"    → {note}")
    if report.missing_critical:
        print(f"  Missing critical   : {', '.join(report.missing_critical)}")
    if report.low_confidence:
        print(f"  Low confidence     : {', '.join(report.low_confidence)}")
    print(f"{'='*60}")
    for fname, fs in report.field_scores.items():
        bar = '█' * int(fs.score * 10)
        print(f"  {fname:<22} {bar:<10} {fs.score:.2f}  {fs.value or '—'}")
    print()