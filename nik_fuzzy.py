"""
NIK structure (16 digits)
--------------------------
  [PP][KK][DD][OB][MM][YY][SSSS]
  PP    – Province code (2 digits, 11–94)
  KK    – City/Regency code (2 digits, 01–99)
  DD    – District code (2 digits, 01–99)
  OB+MM – Day of birth padded; women add 40 (01-31 → 41-71)
    1-2   Province
    3-4   City/Regency
    5-6   District
    7-8   Day of birth (01-31 men; 41-71 women)
    9-10  Month (01-12)
    11-12 Year (2 digits)
    13-16 Sequence (0001-9999)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR confusion tables
# ---------------------------------------------------------------------------

# Characters that OCR commonly returns instead of digits.
# NOTE: 'L' is explicitly added here because it is frequently returned for '1'
# in NIK strings (e.g. "80L1…" where L is a mis-read '1').
OCR_TO_DIGIT: Dict[str, str] = {
    'O': '0', 'Q': '0', 'D': '0',                         # zero look-alikes
    'I': '1', 'l': '1', 'i': '1', '|': '1', '!': '1',    # one look-alikes
    'L': '1',                                               # capital L → 1
    'Z': '2', 'z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8', 'R': '8',
    'g': '9', 'q': '9',
}

# Valid Indonesian province codes (2-digit prefix of NIK)
# Source: Permendagri 72/2019 classification
VALID_PROVINCE_CODES = {
    11, 12, 13, 14, 15, 16, 17, 18, 19,   # Sumatera
    21,                                    # Kep. Riau
    31, 32, 33, 34, 35, 36,               # Jawa (31 = DKI Jakarta)
    51, 52, 53,                            # Bali + Nusa Tenggara
    61, 62, 63, 64, 65,                   # Kalimantan
    71, 72, 73, 74, 75, 76,               # Sulawesi
    81, 82,                               # Maluku
    91, 92,                               # Papua
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class NIKCandidate:
    value:            str            # 16-digit string
    confidence:       float          # 0.0 – 1.0
    source:           str            # "exact", "char_sub", "padded_15", …
    structural_score: float          # NIK spec compliance [0.0 – 1.0]
    original_text:    str            # raw OCR string before processing


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class NIKFuzzyExtractor:
    """
    Generates and ranks NIK candidates from noisy OCR text.

    Usage
    -----
        extractor = NIKFuzzyExtractor()
        candidate = extractor.best_candidate(raw_ocr_text)
        if candidate and candidate.confidence > 0.5:
            nik = candidate.value
    """

    MIN_ACCEPTABLE_CONFIDENCE = 0.30

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def best_candidate(
        self,
        raw_text: str,
        min_confidence: float = 0.30,
    ) -> Optional[NIKCandidate]:
        """Return the highest-confidence NIK candidate, or None."""
        candidates = self.generate_candidates(raw_text)
        if not candidates:
            return None
        best = candidates[0]
        return best if best.confidence >= min_confidence else None

    def generate_candidates(self, raw_text: str) -> List[NIKCandidate]:
        """
        Generate all plausible NIK candidates from raw_text, sorted by
        confidence descending.
        """
        if not raw_text:
            return []

        candidates: List[NIKCandidate] = []

        # ---- Strategy 1: exact digit extraction ----
        exact_digits = re.sub(r'\D', '', raw_text)
        if len(exact_digits) == 16:
            candidates.append(
                self._make_candidate(exact_digits, "exact", raw_text, 1.0)
            )

        # ---- Strategy 2: character substitution then digit extraction ----
        substituted  = self._apply_char_substitution(raw_text)
        sub_digits   = re.sub(r'\D', '', substituted)
        if len(sub_digits) == 16 and sub_digits != exact_digits:
            candidates.append(
                self._make_candidate(sub_digits, "char_sub", raw_text, 0.88)
            )

        # ---- Strategy 3: longest continuous digit run ----
        run = self._longest_digit_run(substituted)
        if 14 <= len(run) <= 16:
            padded    = run.ljust(16, '0') if len(run) < 16 else run
            base_conf = {14: 0.50, 15: 0.72, 16: 0.93}[len(run)]
            if not any(c.value == padded for c in candidates):
                candidates.append(
                    self._make_candidate(
                        padded, f"longest_run_{len(run)}", raw_text, base_conf
                    )
                )

        # ---- Strategy 4: 15-digit reconstruction ----
        working_digits = (
            sub_digits if len(sub_digits) == 15
            else (exact_digits if len(exact_digits) == 15 else None)
        )
        if working_digits:
            candidates.extend(self._reconstruct_from_15(working_digits, raw_text))

        # ---- Validate and rescore ----
        for c in candidates:
            c.structural_score = self._validate_structure(c.value)
            if c.structural_score < 0.2:
                c.confidence *= 0.25
            else:
                c.confidence *= (0.5 + 0.5 * c.structural_score)

        candidates = self._deduplicate(candidates)
        candidates = [
            c for c in candidates if c.confidence >= self.MIN_ACCEPTABLE_CONFIDENCE
        ]
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates

    def extract_from_ocr_items(
        self,
        items: List[dict],
        nik_y_hint: Optional[float] = None,
    ) -> Optional[NIKCandidate]:
        """
        Extract NIK from a list of OCR items (each with 'text', 'box', 'id').

        If nik_y_hint is provided (y-center of the NIK label row), only items
        within 60px vertically are searched first; the full list is used as a
        fallback.
        """
        def _search(subset):
            all_cands = []
            for item in subset:
                all_cands.extend(self.generate_candidates(item['text']))
            all_cands.sort(key=lambda x: x.confidence, reverse=True)
            return all_cands[0] if all_cands else None

        if nik_y_hint is not None:
            near = [
                it for it in items
                if abs((it['box'][0][1] + it['box'][2][1]) / 2 - nik_y_hint) < 60
            ]
            result = _search(near)
            if result and result.confidence >= 0.5:
                return result

        return _search(items)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_char_substitution(self, text: str) -> str:
        """Replace OCR-confused characters with their digit equivalents."""
        return ''.join(OCR_TO_DIGIT.get(ch, ch) for ch in text)

    def _longest_digit_run(self, text: str) -> str:
        """Return the longest consecutive digit-only substring."""
        runs = re.findall(r'\d+', text)
        return max(runs, key=len) if runs else ''

    def _make_candidate(
        self, value: str, source: str, original: str, base_conf: float
    ) -> NIKCandidate:
        return NIKCandidate(
            value=value,
            confidence=base_conf,
            source=source,
            structural_score=0.0,
            original_text=original,
        )

    def _reconstruct_from_15(
        self, digits_15: str, original: str
    ) -> List[NIKCandidate]:
        """
        Generate plausible 16-digit NIK candidates from a 15-digit string.

        High-risk positions (0-indexed in the 16-digit NIK):
          0-1   Province prefix (leading digit drops happen most often)
          6-7   Day of birth
          8-9   Month
          12-15 Sequence (trailing digits drop)
        """
        priority_positions = [0, 1, 6, 7, 8, 9, 12, 13, 14, 15]
        seen   = set()
        result = []

        for pos in priority_positions:
            for digit in '0123456789':
                candidate_str = digits_15[:pos] + digit + digits_15[pos:]
                if candidate_str in seen:
                    continue
                seen.add(candidate_str)
                result.append(
                    self._make_candidate(
                        candidate_str, f"padded_pos{pos}", original, 0.62
                    )
                )

        # Also try edge padding (append / prepend)
        for digit in '0123456789':
            for val in [digit + digits_15, digits_15 + digit]:
                if val not in seen:
                    seen.add(val)
                    result.append(
                        self._make_candidate(val, "edge_pad", original, 0.58)
                    )

        return result

    def _validate_structure(self, nik: str) -> float:
        """
        Score NIK structural validity [0.0 – 1.0].

        Checks:
          * Exactly 16 digits
          * Province code in valid range
          * Day-of-birth plausible (01-31 or 41-71)
          * Month plausible (01-12)
          * Sequence non-zero
        """
        if not nik or not re.match(r'^\d{16}$', nik):
            return 0.0

        score = 1.0

        # Province (digits 1-2)
        prov = int(nik[0:2])
        if prov not in VALID_PROVINCE_CODES:
            if prov < 11 or prov > 94:
                score *= 0.40
            else:
                score *= 0.85

        # District (digits 5-6) must be non-zero
        district = int(nik[4:6])
        if district == 0:
            score *= 0.70

        # Day of birth (digits 7-8)
        day = int(nik[6:8])
        if day == 0:
            score *= 0.0
        elif 1 <= day <= 31:
            pass           # male pattern
        elif 41 <= day <= 71:
            pass           # female pattern
        else:
            score *= 0.10  # 32-40 or 72+ → very suspicious

        # Month (digits 9-10)
        month = int(nik[8:10])
        if month < 1 or month > 12:
            score *= 0.0

        # Sequence (digits 13-16)
        seq = int(nik[12:16])
        if seq == 0:
            score *= 0.50

        return float(score)

    def _deduplicate(self, candidates: List[NIKCandidate]) -> List[NIKCandidate]:
        """Keep only the highest-confidence entry for each unique value."""
        seen: Dict[str, NIKCandidate] = {}
        for c in candidates:
            if c.value not in seen or c.confidence > seen[c.value].confidence:
                seen[c.value] = c
        return list(seen.values())


# ---------------------------------------------------------------------------
# Stand-alone utility
# ---------------------------------------------------------------------------

def clean_nik_robust(raw: str, min_confidence: float = 0.30) -> Optional[str]:
    """
    Drop-in upgrade for the existing _clean_nik() helper.
    Returns a 16-digit string or None.

    Usage
    -----
        from nik_fuzzy import clean_nik_robust
        nik = clean_nik_robust(raw_text)
    """
    extractor = NIKFuzzyExtractor()
    candidate = extractor.best_candidate(raw, min_confidence=min_confidence)
    if candidate:
        logger.debug(
            "NIK extracted: %s | conf=%.3f | source=%s",
            candidate.value, candidate.confidence, candidate.source,
        )
        return candidate.value
    return None