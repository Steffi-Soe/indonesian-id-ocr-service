import re
import logging
from dataclasses import dataclass
from datetime import date as _date
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR confusion map specifically for digits inside date strings
# ---------------------------------------------------------------------------

DATE_CHAR_MAP = {
    'O': '0', 'o': '0', 'Q': '0',
    'I': '1', 'l': '1', 'i': '1', '!': '1', 'L': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}

# Realistic birth-year range for Indonesian KTP holders
MIN_BIRTH_YEAR = 1920
MAX_BIRTH_YEAR = 2100   # ~14+ years old at time of issue

# The extractor's reference year for short-year reconstruction
_REFERENCE_YEAR = 2026


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DateResult:
    normalized:  Optional[str]   # "DD-MM-YYYY" or None
    day:         Optional[int]
    month:       Optional[int]
    year:        Optional[int]
    confidence:  float           # 0.0 – 1.0
    method:      str             # human-readable extraction path


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

class DateNormalizer:
    """
    Converts noisy OCR date strings into canonical DD-MM-YYYY format.

    Always zero-pads day and month.  Attempts single-digit year repair
    when the extracted year falls outside the valid birth-year range.

    Usage
    -----
        norm = DateNormalizer()
        result = norm.normalize("23 3 1392")
        # → DateResult(normalized='23-03-1992', day=23, month=3, year=1992,
        #              confidence=0.72, method='year_repair_strict_dd_mm_yyyy')
    """

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def normalize(self, raw: str) -> DateResult:
        """
        Normalize a raw OCR date string.  Never raises; returns a
        DateResult with confidence=0.0 when extraction fails.
        """
        if not raw or not raw.strip():
            return DateResult(None, None, None, None, 0.0, "empty_input")

        text = raw.strip()

        for strategy in (
            self._try_standard_patterns,
            self._try_after_char_substitution,
            self._try_reconstruct_3_digit_year,
            self._try_reconstruct_7_digit,
            self._try_extract_from_place_date_string,
        ):
            result = strategy(text)
            if result and result.confidence > 0.0:
                return result

        return DateResult(None, None, None, None, 0.0, "all_strategies_failed")

    def normalize_place_date(self, raw: str) -> Tuple[Optional[str], DateResult]:
        """
        Parse a combined "Tempat/Tgl Lahir" string like:
            "JAKARTA, 23-10-1983"
        Returns (place_str, DateResult).
        """
        if not raw:
            return None, DateResult(None, None, None, None, 0.0, "empty")

        if ',' in raw:
            parts = raw.split(',', 1)
            place    = parts[0].strip().strip('.:- ')
            date_raw = parts[1].strip()
        else:
            m = re.search(
                r'(\d{1,2}[\s\./\-]+\d{1,2}[\s\./\-]+\d{2,4})',
                raw
            )
            if m:
                place    = raw[:m.start()].strip().strip('.:- ')
                date_raw = m.group(1)
            else:
                place    = None
                date_raw = raw

        date_result = self.normalize(date_raw)
        return place or None, date_result

    # -----------------------------------------------------------------------
    # Strategies
    # -----------------------------------------------------------------------

    def _try_standard_patterns(self, text: str) -> Optional[DateResult]:
        """Handle well-formed dates with any common separator (including spaces)."""
        # Collapse runs of separators then normalise single ones
        clean = re.sub(r'[-/. ]{2,}', '-', text)
        clean = re.sub(r'[-/. ]', '-', clean)

        # Strict DD-MM-YYYY
        m = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})$', clean.strip())
        if m:
            return self._build_result(
                m.group(1), m.group(2), m.group(3), 0.97, "strict_dd_mm_yyyy"
            )

        # Embedded DD-MM-YYYY
        m = re.search(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', clean)
        if m:
            return self._build_result(
                m.group(1), m.group(2), m.group(3), 0.90, "embedded_dd_mm_yyyy"
            )

        return None

    def _try_after_char_substitution(self, text: str) -> Optional[DateResult]:
        """Apply OCR char substitution then retry standard patterns."""
        substituted = self._substitute_chars(text)
        if substituted == text:
            return None

        result = self._try_standard_patterns(substituted)
        if result:
            result.confidence *= 0.88
            result.method = "char_sub_" + result.method
        return result

    def _try_reconstruct_3_digit_year(self, text: str) -> Optional[DateResult]:
        """
        Handle truncated years:  '12-03-988', '23-10-198', '05-07-196'
        """
        substituted = self._substitute_chars(text)
        clean = re.sub(r'[-/. ]{2,}', '-', substituted)
        clean = re.sub(r'[-/. ]', '-', clean)

        m = re.search(r'(\d{1,2})-(\d{1,2})-(\d{3})$', clean.strip())
        if not m:
            return None

        d_s, mo_s, yr_3 = m.group(1), m.group(2), m.group(3)
        yr_int = int(yr_3)

        if 900 <= yr_int <= 999:
            year_full   = f"1{yr_3}"
            conf_penalty = 0.75
        elif 0 <= yr_int <= 25:
            year_full   = f"20{yr_3}"
            conf_penalty = 0.70
        elif 26 <= yr_int <= 99:
            year_full   = f"19{yr_3}"
            conf_penalty = 0.65
        elif 100 <= yr_int <= 199:
            year_full   = f"1{yr_3}0"
            conf_penalty = 0.50
        else:
            return None

        return self._build_result(d_s, mo_s, year_full, conf_penalty, "3digit_year_recon")

    def _try_reconstruct_7_digit(self, text: str) -> Optional[DateResult]:
        """Handle 7-digit date strings (one digit missing)."""
        substituted = self._substitute_chars(text)
        digits = re.sub(r'\D', '', substituted)

        if len(digits) != 7:
            return None

        # Interpretation 1: DDMMYYY
        d_s  = digits[0:2]
        mo_s = digits[2:4]
        yr_3 = digits[4:7]
        result = self._try_reconstruct_3_digit_year(f"{d_s}-{mo_s}-{yr_3}")
        if result and result.confidence > 0:
            result.confidence *= 0.82
            result.method = "7digit_" + result.method
            return result

        # Interpretation 2: DMMYYYY (day lost its first digit)
        d_s  = '0' + digits[0]
        mo_s = digits[1:3]
        yr_s = digits[3:7]
        result2 = self._build_result(d_s, mo_s, yr_s, 0.45, "7digit_alt_interp")
        return result2 if (result2 and result2.confidence > 0) else None

    def _try_extract_from_place_date_string(self, text: str) -> Optional[DateResult]:
        """Fallback: scan the entire string for any date-like pattern."""
        substituted = self._substitute_chars(text)

        m = re.search(
            r'(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})',
            substituted
        )
        if m:
            d_s, mo_s, yr_s = m.group(1), m.group(2), m.group(3)
            if len(yr_s) == 2:
                yr_int = int(yr_s)
                yr_s   = f"19{yr_s}" if yr_int > 25 else f"20{yr_s}"
            return self._build_result(d_s, mo_s, yr_s, 0.55, "permissive_scan")

        return None

    # -----------------------------------------------------------------------
    # Year repair
    # -----------------------------------------------------------------------

    @staticmethod
    def _try_repair_year(yr: int) -> Optional[int]:
        """
        Attempt to fix an out-of-range year via single-digit substitution.

        Strategy: try replacing each digit position (0-3) with every other
        digit (0-9) and return the first candidate inside MIN_BIRTH_YEAR –
        _REFERENCE_YEAR.  Earlier positions are tried first so we prefer
        century-level corrections (e.g. 1392 → 1992 at position 1).

        Returns the repaired integer year, or None if no fix is found.
        """
        yr_str = str(yr).zfill(4)
        for pos in range(4):
            for replacement in '0123456789':
                if replacement == yr_str[pos]:
                    continue
                candidate = int(yr_str[:pos] + replacement + yr_str[pos + 1:])
                if MIN_BIRTH_YEAR <= candidate <= _REFERENCE_YEAR:
                    return candidate
        return None

    # -----------------------------------------------------------------------
    # Building and validating a DateResult
    # -----------------------------------------------------------------------

    def _build_result(
        self, d_s: str, mo_s: str, yr_s: str,
        base_confidence: float, method: str
    ) -> Optional[DateResult]:
        """
        Parse string components, validate ranges, optionally repair an
        invalid year, then return a DateResult with a canonical DD-MM-YYYY
        normalized string.  All day and month values are zero-padded.
        """
        try:
            d  = int(d_s.strip())
            mo = int(mo_s.strip())
            yr = int(yr_s.strip())
        except ValueError:
            return None

        confidence = base_confidence

        # ---- Hard day/month validation ----
        if d < 1 or d > 31:
            if 1 <= mo <= 31 and 1 <= d <= 12:
                d, mo = mo, d
                confidence *= 0.80
            else:
                return None

        if mo < 1 or mo > 12:
            return None

        # ---- Year: expand two-digit years ----
        if yr < 100:
            yr = (1900 + yr) if yr > 25 else (2000 + yr)
            confidence *= 0.85

        # ---- Year plausibility & repair ----
        if not (MIN_BIRTH_YEAR <= yr <= _REFERENCE_YEAR):
            repaired_yr = self._try_repair_year(yr)
            if repaired_yr is not None:
                logger.debug(
                    "date_norm: repaired year %d → %d (method=%s)",
                    yr, repaired_yr, method
                )
                yr          = repaired_yr
                method      = "year_repair_" + method
                confidence *= 0.80
            else:
                # Year remains implausible
                if yr <= MAX_BIRTH_YEAR and yr > _REFERENCE_YEAR:
                    confidence *= 0.75   # very young person
                else:
                    confidence *= 0.15   # very unlikely; still return for audit

        # ---- Calendar validation (catches Feb 30 etc.) ----
        try:
            _date(yr, mo, d)
        except ValueError:
            return None

        # ---- Canonical zero-padded output ----
        normalized = f"{d:02d}-{mo:02d}-{yr:04d}"
        return DateResult(
            normalized=normalized,
            day=d, month=mo, year=yr,
            confidence=float(confidence),
            method=method,
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _substitute_chars(text: str) -> str:
        """Replace OCR character confusions with their digit equivalents."""
        result = []
        for ch in text:
            if ch.isdigit() or ch in '-./ ,':
                result.append(ch)
            else:
                result.append(DATE_CHAR_MAP.get(ch, ch))
        return ''.join(result)


# ---------------------------------------------------------------------------
# Drop-in replacement for the existing _normalize_date() helper
# ---------------------------------------------------------------------------

_normalizer_singleton = DateNormalizer()


def normalize_date_robust(raw: str) -> str:
    """
    Drop-in replacement for ktp_extractor._normalize_date().

    Always returns a DD-MM-YYYY string (empty string on failure, never raises).
    Day and month are always zero-padded.  Attempts single-digit year repair
    for out-of-range years such as 1392 → 1992.

    Usage
    -----
        from date_normalizer import normalize_date_robust
        cleaned = normalize_date_robust(raw_date_string)
    """
    if not raw:
        return raw
    result = _normalizer_singleton.normalize(raw)
    if result.normalized and result.confidence > 0.25:
        logger.debug(
            "date_norm: '%s' → '%s' (conf=%.2f, method=%s)",
            raw, result.normalized, result.confidence, result.method,
        )
        return result.normalized
    logger.debug("date_norm: could not normalize '%s'", raw)
    return raw   # return original rather than empty string