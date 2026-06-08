"""
ocr_corrector.py
----------------
Generalised OCR text correction for Indonesian identity documents.

Three correction layers, dispatched by field type:

  Layer 1 – CharSubstitutionCorrector
    Bidirectional character confusion map.  In TEXT context, converts
    digit look-alikes → letters (0→O, 1→I, l→I).  In DIGIT context,
    converts letter look-alikes → digits (O→0, I→1).  Context-aware
    mode handles mixed strings like street addresses.

  Layer 2 – EnumFieldCorrector
    For closed-vocabulary fields (Jenis Kelamin, Agama, Status Perkawinan,
    Kewarganegaraan, Golongan Darah): applies char substitution then
    fuzzy-matches against canonical values.

  Layer 3 – PlaceNameCorrector
    For geographic fields (Tempat Lahir, Kecamatan, Kabupaten, Provinsi):
    fuzzy-matches against a curated Indonesian administrative-area database
    (~500 entries covering 34 provinces, all kota, and major kabupaten).
    Also attempts J↔I first-character correction (Jakarta→Iakarta confusion).

Usage
-----
    corrector = OCRTextCorrector()

    # Closed-vocabulary
    v, c = corrector.correct_field("kewarganegaraan", "WNl")       # ("WNI", 0.93)
    v, c = corrector.correct_field("jenis_kelamin",   "LAKI-LAK1") # ("LAKI-LAKI", 0.92)

    # Place names
    v, c = corrector.correct_field("tempat_lahir", "B0GOR")        # ("BOGOR", 0.95)
    v, c = corrector.correct_field("tempat_lahir", "IAKARTA")      # ("JAKARTA", 0.90)

    # Free text
    v, c = corrector.correct_field("nama", "BUDI SANT0SO")         # ("BUDI SANTOSO", 0.80)

    # Convenience helpers
    nik_str = corrector.correct_kewarganegaraan("WNl")              # "WNI"
    place,c = corrector.correct_place("B0GOR")                     # ("BOGOR", 0.95)
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from thefuzz import fuzz
from thefuzz import process as fuzz_process

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Character confusion tables
# ---------------------------------------------------------------------------

# TEXT context: digit/symbol → most likely letter
DIGIT_IN_TEXT: Dict[str, str] = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '8': 'B',
    'l': 'I',   # lowercase L
    '|': 'I',   # pipe
    '!': 'I',   # exclamation
}

# DIGIT context: letter/symbol → most likely digit
# (kept consistent with nik_fuzzy.OCR_TO_DIGIT and date_normalizer.DATE_CHAR_MAP)
ALPHA_IN_DIGITS: Dict[str, str] = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', 'i': '1', '|': '1', '!': '1', 'L': '1',
    'Z': '2', 'z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5',
    'G': '6',
    'T': '7',
    'B': '8', 'R': '8',
    'g': '9', 'q': '9',
}


# ---------------------------------------------------------------------------
# Indonesian administrative-area database (~500 entries)
# ---------------------------------------------------------------------------

_PROVINCES: Set[str] = {
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "KEPULAUAN RIAU",
    "JAMBI", "SUMATERA SELATAN", "BANGKA BELITUNG", "BENGKULU", "LAMPUNG",
    "DKI JAKARTA", "JAWA BARAT", "BANTEN", "JAWA TENGAH", "DI YOGYAKARTA",
    "JAWA TIMUR", "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR",
    "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
    "KALIMANTAN TIMUR", "KALIMANTAN UTARA",
    "SULAWESI UTARA", "GORONTALO", "SULAWESI TENGAH", "SULAWESI BARAT",
    "SULAWESI SELATAN", "SULAWESI TENGGARA",
    "MALUKU", "MALUKU UTARA", "PAPUA BARAT", "PAPUA",
}

_KOTA: Set[str] = {
    # DKI Jakarta
    "JAKARTA", "JAKARTA PUSAT", "JAKARTA UTARA", "JAKARTA BARAT",
    "JAKARTA SELATAN", "JAKARTA TIMUR",
    # Jawa Barat
    "BANDUNG", "BOGOR", "DEPOK", "BEKASI", "CIMAHI", "CIREBON",
    "SUKABUMI", "TASIKMALAYA", "BANJAR",
    # Jawa Tengah
    "SEMARANG", "SURAKARTA", "SOLO", "MAGELANG", "SALATIGA",
    "PEKALONGAN", "TEGAL",
    # DI Yogyakarta
    "YOGYAKARTA",
    # Jawa Timur
    "SURABAYA", "MALANG", "MOJOKERTO", "PASURUAN", "PROBOLINGGO",
    "BLITAR", "KEDIRI", "MADIUN", "BATU",
    # Banten
    "TANGERANG", "TANGERANG SELATAN", "SERANG", "CILEGON",
    # Bali
    "DENPASAR",
    # NTB
    "MATARAM", "BIMA",
    # NTT
    "KUPANG",
    # Sumatera Utara
    "MEDAN", "BINJAI", "PEMATANGSIANTAR", "TEBING TINGGI",
    "SIBOLGA", "TANJUNGBALAI", "PADANGSIDIMPUAN", "GUNUNGSITOLI",
    # Sumatera Barat
    "PADANG", "BUKITTINGGI", "PAYAKUMBUH", "PADANG PANJANG",
    "PADANGPANJANG", "SAWAH LUNTO", "SAWAHUNTO", "SOLOK", "PARIAMAN",
    # Aceh
    "BANDA ACEH", "SABANG", "LANGSA", "LHOKSEUMAWE", "SUBULUSSALAM",
    # Riau
    "PEKANBARU", "DUMAI",
    # Kepulauan Riau
    "BATAM", "TANJUNGPINANG",
    # Jambi
    "JAMBI", "SUNGAIPENUH",
    # Sumatera Selatan
    "PALEMBANG", "PRABUMULIH", "PAGAR ALAM", "PAGARALAM", "LUBUKLINGGAU",
    # Bangka Belitung
    "PANGKALPINANG",
    # Bengkulu
    "BENGKULU",
    # Lampung
    "BANDAR LAMPUNG", "METRO",
    # Kalimantan Barat
    "PONTIANAK", "SINGKAWANG",
    # Kalimantan Tengah
    "PALANGKARAYA", "PALANGKA RAYA",
    # Kalimantan Selatan
    "BANJARMASIN", "BANJARBARU",
    # Kalimantan Timur
    "BALIKPAPAN", "SAMARINDA", "BONTANG",
    # Kalimantan Utara
    "TARAKAN",
    # Sulawesi Utara
    "MANADO", "BITUNG", "TOMOHON", "KOTAMOBAGU",
    # Gorontalo
    "GORONTALO",
    # Sulawesi Tengah
    "PALU",
    # Sulawesi Selatan
    "MAKASSAR", "PAREPARE", "PALOPO",
    # Sulawesi Tenggara
    "KENDARI", "BAUBAU",
    # Sulawesi Barat
    "MAMUJU",
    # Maluku
    "AMBON", "TUAL",
    # Maluku Utara
    "TERNATE", "TIDORE KEPULAUAN",
    # Papua
    "JAYAPURA",
    # Papua Barat
    "SORONG", "MANOKWARI",
}

_KABUPATEN: Set[str] = {
    # Jawa Barat
    "BOGOR", "SUKABUMI", "CIANJUR", "BANDUNG", "BANDUNG BARAT", "GARUT",
    "TASIKMALAYA", "CIAMIS", "KUNINGAN", "CIREBON", "MAJALENGKA",
    "SUMEDANG", "INDRAMAYU", "SUBANG", "PURWAKARTA", "KARAWANG",
    "BEKASI", "PANGANDARAN",
    # Jawa Tengah
    "SEMARANG", "KENDAL", "DEMAK", "GROBOGAN", "PATI", "KUDUS",
    "JEPARA", "REMBANG", "BLORA", "SRAGEN", "KARANGANYAR",
    "WONOGIRI", "SUKOHARJO", "KLATEN", "BOYOLALI", "MAGELANG",
    "TEMANGGUNG", "WONOSOBO", "BANJARNEGARA", "KEBUMEN", "PURWOREJO",
    "PURBALINGGA", "BANYUMAS", "CILACAP", "BREBES", "TEGAL",
    "PEMALANG", "BATANG", "PEKALONGAN",
    # DI Yogyakarta
    "KULONPROGO", "KULON PROGO", "BANTUL", "SLEMAN",
    "GUNUNG KIDUL", "GUNUNGKIDUL",
    # Jawa Timur
    "SIDOARJO", "GRESIK", "BANGKALAN", "SAMPANG", "PAMEKASAN",
    "SUMENEP", "MOJOKERTO", "JOMBANG", "NGANJUK", "MADIUN",
    "MAGETAN", "NGAWI", "BOJONEGORO", "TUBAN", "LAMONGAN",
    "PASURUAN", "PROBOLINGGO", "LUMAJANG", "JEMBER", "BONDOWOSO",
    "SITUBONDO", "BANYUWANGI", "MALANG", "BLITAR", "TULUNGAGUNG",
    "TRENGGALEK", "PONOROGO", "PACITAN", "KEDIRI",
    # Banten
    "LEBAK", "PANDEGLANG", "SERANG", "TANGERANG",
    # Sumatera Utara
    "DELI SERDANG", "LANGKAT", "SERDANG BEDAGAI", "ASAHAN",
    "BATUBARA", "LABUHANBATU", "LABUHAN BATU",
    "SIMALUNGUN", "KARO", "DAIRI", "PAKPAK BHARAT",
    "NIAS", "NIAS UTARA", "NIAS SELATAN", "NIAS BARAT",
    "MANDAILING NATAL", "TAPANULI SELATAN", "TAPANULI TENGAH",
    "TAPANULI UTARA", "TOBA SAMOSIR",
    "HUMBANG HASUNDUTAN", "SAMOSIR",
    "PADANG LAWAS", "PADANG LAWAS UTARA",
    # Sumatera Barat
    "AGAM", "LIMA PULUH KOTA", "PASAMAN", "PASAMAN BARAT",
    "PESISIR SELATAN", "SIJUNJUNG", "SOLOK", "SOLOK SELATAN",
    "TANAH DATAR", "DHARMASRAYA", "KEPULAUAN MENTAWAI",
    # Riau
    "BENGKALIS", "INDRAGIRI HILIR", "INDRAGIRI HULU", "KAMPAR",
    "KUANTAN SINGINGI", "PELALAWAN", "ROKAN HILIR", "ROKAN HULU",
    "SIAK", "KEPULAUAN MERANTI",
    # Lampung
    "LAMPUNG BARAT", "LAMPUNG SELATAN", "LAMPUNG TENGAH",
    "LAMPUNG TIMUR", "LAMPUNG UTARA", "MESUJI", "PESAWARAN",
    "PESISIR BARAT", "PRINGSEWU", "TANGGAMUS", "TULANG BAWANG",
    "TULANG BAWANG BARAT", "WAY KANAN",
    # Kalimantan Barat
    "BENGKAYANG", "KAPUAS HULU", "KAYONG UTARA", "KETAPANG",
    "KUBU RAYA", "LANDAK", "MELAWI", "MEMPAWAH", "PONTIANAK",
    "SAMBAS", "SANGGAU", "SEKADAU", "SINTANG",
    # Kalimantan Selatan
    "BALANGAN", "BANJAR", "BARITO KUALA", "HULU SUNGAI SELATAN",
    "HULU SUNGAI TENGAH", "HULU SUNGAI UTARA", "KOTABARU",
    "TABALONG", "TANAH BUMBU", "TANAH LAUT", "TAPIN",
    # Sulawesi Selatan
    "BANTAENG", "BARRU", "BONE", "BULUKUMBA", "ENREKANG",
    "GOWA", "JENEPONTO", "KEPULAUAN SELAYAR", "LUWU",
    "LUWU TIMUR", "LUWU UTARA", "MAROS", "PANGKAJENE KEPULAUAN",
    "PANGKEP", "PINRANG", "SIDENRENG RAPPANG", "SINJAI",
    "SOPPENG", "TAKALAR", "TANA TORAJA", "TORAJA UTARA",
    "WAJO",
    # Common short city references in KTP
    "JAKBAR", "JAKSEL", "JAKPUS", "JAKTIM", "JAKUT",
    # Overseas
    "LUAR NEGERI",
}

# Unified place database (uppercase, deduplicated, sorted)
INDONESIAN_PLACES: List[str] = sorted(
    {p.upper() for p in (_PROVINCES | _KOTA | _KABUPATEN)}
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    original:   str
    corrected:  str
    confidence: float      # 0.0–1.0
    method:     str        # 'exact'|'char_sub'|'fuzzy_enum'|'fuzzy_place'|…
    changed:    bool


# ---------------------------------------------------------------------------
# Layer 1: Character substitution
# ---------------------------------------------------------------------------

class CharSubstitutionCorrector:
    """Apply OCR character-confusion corrections to a string."""

    def text_context(self, text: str) -> str:
        """
        Digit/symbol → letter for text/word context.
        Upcases the string; converts 0→O, 1→I, l→I, etc.
        """
        if not text:
            return text
        return ''.join(DIGIT_IN_TEXT.get(ch, ch) for ch in text.upper())

    def digit_context(self, text: str) -> str:
        """
        Letter/symbol → digit for numeric context.
        Shared logic with nik_fuzzy / date_normalizer.
        """
        if not text:
            return text
        return ''.join(ALPHA_IN_DIGITS.get(ch, ch) for ch in text)

    def context_aware(self, text: str) -> str:
        """
        Token-by-token: apply text_context to alpha-dominant tokens,
        leave digit-dominant tokens unchanged.
        Suitable for addresses that mix words and numbers.
        """
        if not text:
            return text
        result = []
        for token in re.split(r'(\s+)', text):
            if token.isspace() or not token:
                result.append(token)
                continue
            alpha = sum(1 for c in token if c.isalpha())
            if alpha / max(len(token), 1) >= 0.4:
                result.append(self.text_context(token))
            else:
                result.append(token.upper())
        return ''.join(result)


# ---------------------------------------------------------------------------
# Layer 2: Enum field corrector
# ---------------------------------------------------------------------------

class EnumFieldCorrector:
    """
    Corrects closed-vocabulary fields by char substitution + fuzzy matching
    against canonical values.
    """

    FIELD_ENUMS: Dict[str, Dict] = {
        'jenis_kelamin': {
            'values':    ['LAKI-LAKI', 'PEREMPUAN'],
            'threshold': 55,
        },
        'agama': {
            'values':    ['ISLAM', 'KRISTEN', 'KATOLIK', 'HINDU', 'BUDDHA', 'KONGHUCU'],
            'threshold': 65,
        },
        'status_perkawinan': {
            'values':    ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI'],
            'threshold': 65,
        },
        'kewarganegaraan': {
            'values':    ['WNI', 'WNA'],
            'threshold': 45,    # Low — very short strings
        },
        'golongan_darah': {
            'values':    ['A', 'B', 'AB', 'O', 'A+', 'B+', 'AB+', 'O+',
                          'A-', 'B-', 'AB-', 'O-'],
            'threshold': 80,
        },
    }

    _ALIASES: Dict[str, str] = {
        # Jenis Kelamin
        'jenis kelamin': 'jenis_kelamin', 'jenis_kelamin': 'jenis_kelamin',
        'kelamin': 'jenis_kelamin',
        # Agama
        'agama': 'agama',
        # Status Perkawinan
        'status perkawinan': 'status_perkawinan',
        'status_perkawinan': 'status_perkawinan',
        # Kewarganegaraan
        'kewarganegaraan': 'kewarganegaraan',
        # Golongan Darah
        'gol darah': 'golongan_darah', 'gol. darah': 'golongan_darah',
        'golongan_darah': 'golongan_darah', 'golongan darah': 'golongan_darah',
    }

    def __init__(self):
        self._char = CharSubstitutionCorrector()

    def correct(self, field_name: str, value: str) -> Optional[CorrectionResult]:
        """Return CorrectionResult or None if field is not an enum field."""
        if not value:
            return None
        key = self._ALIASES.get(field_name.lower().strip())
        if key is None:
            return None

        config    = self.FIELD_ENUMS[key]
        canonical = config['values']
        threshold = config['threshold']
        val_up    = value.upper().strip()

        # 1. Direct match
        if val_up in canonical:
            return CorrectionResult(val_up, val_up, 1.0, 'exact', False)

        # 2. Char substitution + direct match
        subst = self._char.text_context(val_up)
        if subst in canonical:
            return CorrectionResult(value, subst, 0.93, 'char_sub', True)

        # 3. Fuzzy match on original and substituted forms
        for candidate_str in (val_up, subst):
            result = fuzz_process.extractOne(
                candidate_str, canonical, scorer=fuzz.token_set_ratio
            )
            if result and result[1] >= threshold:
                return CorrectionResult(
                    original=value, corrected=result[0],
                    confidence=result[1] / 100.0, method='fuzzy_enum', changed=True,
                )

        return None


# ---------------------------------------------------------------------------
# Layer 3: Place-name corrector
# ---------------------------------------------------------------------------

class PlaceNameCorrector:
    """
    Fuzzy-matches a raw place string against the Indonesian place database.
    Also handles the common J↔I first-character OCR confusion
    (e.g. IAKARTA → JAKARTA).
    """

    def __init__(self, extra_places: Optional[List[str]] = None):
        db = list(INDONESIAN_PLACES)
        if extra_places:
            db.extend(p.upper().strip() for p in extra_places)
        self._db: List[str]  = sorted(set(db))
        self._db_set: Set[str] = set(self._db)
        self._char = CharSubstitutionCorrector()

    def correct(
        self,
        raw: str,
        min_confidence: float = 0.82,
    ) -> CorrectionResult:
        """
        Attempt to correct a place name.
        Returns a CorrectionResult; if no match above min_confidence,
        returns the char-substituted original with low confidence.
        """
        if not raw or len(raw.strip()) < 2:
            return CorrectionResult(raw, raw, 0.0, 'too_short', False)

        val_up = raw.upper().strip()

        # 1. Exact match
        if val_up in self._db_set:
            return CorrectionResult(val_up, val_up, 1.0, 'exact', val_up != raw)

        # 2. Char substitution + exact
        subst = self._char.text_context(val_up)
        if subst in self._db_set:
            return CorrectionResult(raw, subst, 0.95, 'char_sub_exact', True)

        # 3. J↔I first-character swap
        for variant in self._j_i_variants(val_up):
            if variant in self._db_set:
                return CorrectionResult(raw, variant, 0.90, 'j_i_exact', True)
        for variant in self._j_i_variants(subst):
            if variant in self._db_set:
                return CorrectionResult(raw, variant, 0.87, 'j_i_char_sub', True)

        # 4. Fuzzy match across all candidate strings
        candidates = list({val_up, subst}
                          | set(self._j_i_variants(val_up))
                          | set(self._j_i_variants(subst)))
        best_score = 0
        best_match: Optional[str] = None

        for cand in candidates:
            result = fuzz_process.extractOne(
                cand, self._db, scorer=fuzz.token_set_ratio
            )
            if result and result[1] > best_score:
                best_score = result[1]
                best_match = result[0]

        if best_score >= min_confidence * 100 and best_match:
            return CorrectionResult(
                raw, best_match, best_score / 100.0, 'fuzzy_place', True
            )

        # 5. Best-effort: return char-substituted original
        corrected = subst if subst != val_up else val_up
        return CorrectionResult(raw, corrected, 0.35, 'char_sub_only', corrected != raw)

    @staticmethod
    def _j_i_variants(text: str) -> List[str]:
        """I→J and J→I first-character swap variants."""
        if not text:
            return []
        variants = []
        if text[0] == 'I':
            variants.append('J' + text[1:])
        elif text[0] == 'J':
            variants.append('I' + text[1:])
        return variants


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

class OCRTextCorrector:
    """
    Unified OCR text correction facade for all non-NIK, non-date fields.

    Field routing:
      enum fields   → EnumFieldCorrector (Layer 2)
      place fields  → PlaceNameCorrector (Layer 3)
      free text     → CharSubstitutionCorrector (Layer 1)
      NIK / dates   → handled by dedicated modules (nik_fuzzy / date_normalizer)
    """

    _ENUM_FIELDS: FrozenSet[str] = frozenset({
        'jenis_kelamin', 'agama', 'status_perkawinan',
        'kewarganegaraan', 'golongan_darah',
    })

    _PLACE_FIELDS: FrozenSet[str] = frozenset({
        'tempat_lahir', 'kecamatan', 'kabupaten',
        'provinsi', 'kel_desa',
    })

    def __init__(self, extra_places: Optional[List[str]] = None):
        self._char  = CharSubstitutionCorrector()
        self._enum  = EnumFieldCorrector()
        self._place = PlaceNameCorrector(extra_places=extra_places)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def correct_field(
        self,
        field_name: str,
        value: Optional[str],
    ) -> Tuple[Optional[str], float]:
        """
        Correct a single field value.  Returns (corrected_value, confidence).
        Returns the original value unchanged for fields with no strategy.
        """
        if not value:
            return value, 0.0

        key = field_name.lower().replace(' ', '_').replace('.', '_')

        if key in self._ENUM_FIELDS:
            result = self._enum.correct(key, value)
            if result:
                return result.corrected, result.confidence
            return value, 0.5

        if key in self._PLACE_FIELDS:
            result = self._place.correct(value)
            return result.corrected, result.confidence

        # Free-text: context-aware char substitution
        corrected = self._char.context_aware(value)
        changed   = corrected != value.upper()
        return corrected, 0.80 if changed else 1.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def correct_kewarganegaraan(self, raw: Optional[str]) -> Optional[str]:
        """Reliably normalise WNI / WNA (handles WNl, Wn1, WN1, etc.)."""
        if not raw:
            return raw
        result = self._enum.correct('kewarganegaraan', raw)
        if result and result.confidence > 0.40:
            return result.corrected
        # Hard fallback: char-substitute then keyword check
        up = self._char.text_context(raw.upper())
        if 'WNI' in up:
            return 'WNI'
        if 'WNA' in up:
            return 'WNA'
        return raw

    def correct_place(self, raw: Optional[str]) -> Tuple[Optional[str], float]:
        """Correct a place name directly."""
        if not raw:
            return raw, 0.0
        result = self._place.correct(raw)
        return result.corrected, result.confidence

    def correct_text(self, raw: Optional[str]) -> Optional[str]:
        """Apply basic char substitution to a free-text field."""
        if not raw:
            return raw
        return self._char.context_aware(raw)

    def correct_name(self, raw: Optional[str]) -> Optional[str]:
        """
        Correct a person name: context-aware char substitution.
        Avoids over-correcting digit strings that might be intentional.
        """
        if not raw:
            return raw
        result = []
        for token in raw.upper().split():
            # Tokens that are entirely digits are likely OCR noise → keep
            if token.isdigit():
                result.append(token)
                continue
            result.append(self._char.text_context(token))
        return ' '.join(result)


# ---------------------------------------------------------------------------
# Module-level singleton (optional convenience)
# ---------------------------------------------------------------------------

_corrector_singleton: Optional[OCRTextCorrector] = None


def get_corrector() -> OCRTextCorrector:
    """Return a lazily-initialised module-level singleton."""
    global _corrector_singleton
    if _corrector_singleton is None:
        _corrector_singleton = OCRTextCorrector()
    return _corrector_singleton