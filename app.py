"""SPHEREx SkyBlink v6.0 — Single-File Edition (Public Time-Domain & 102-Channel Near-Infrared Observatory)."""

import base64
import io
import json
import math
import tarfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from reproject import reproject_interp
from requests.adapters import HTTPAdapter
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="SPHEREx SkyBlink | NASA Time-Domain & 102-Channel Observatory",
    page_icon="🔭",
    layout="wide",
)

# =============================================================================
# 1. Core Constants, NASA Targets & SPHEREx 6-Array / 102-Channel Architecture
# =============================================================================
IRSA_TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
IRSA_IBE_CUTOUT_URL = "https://irsa.ipac.caltech.edu/ibe/cutout"
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
CADC_TAP_SYNC_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
UNWISE_CUTOUT_URL = "https://unwise.me/cutout_fits"

SPHEREX_COLLECTIONS = [
    "all",
    "spherex_dr1",
    "spherex_qr3",
    "spherex_dr1_deep",
    "spherex_qr3_deep",
    "spherex_dr1_cal",
]

NASA_BENCHMARK_TARGETS = {
    "Barnard's Star": {
        "ra": "17h57m48.498s",
        "dec": "+04d41m36.21s",
        "ra_deg": 269.452075,
        "dec_deg": 4.693392,
        "category": "High Proper Motion Star (~10.36 arcsec/yr)",
    },
    "Epsilon Indi B": {
        "ra": "22h04m10.520s",
        "dec": "-56d46m57.70s",
        "ra_deg": 331.043833,
        "dec_deg": -56.782694,
        "category": "Binary Brown Dwarf Pair (Ba/Bb, ~4.7 arcsec/yr)",
    },
    "WISE 0855": {
        "ra": "08h55m10.831s",
        "dec": "-07d14m42.53s",
        "ra_deg": 133.795129,
        "dec_deg": -7.245147,
        "category": "Coldest Known Y-Dwarf (~250 K, ~8.1 arcsec/yr)",
    },
    "RW Aurigae": {
        "ra": "05h07m49.538s",
        "dec": "+30d24m05.07s",
        "ra_deg": 76.956408,
        "dec_deg": 30.401408,
        "category": "Variable Pre-Main-Sequence Binary (T Tauri)",
    },
    "R Draconis": {
        "ra": "16h32m40.230s",
        "dec": "+66d45m17.90s",
        "ra_deg": 248.167625,
        "dec_deg": 66.754972,
        "category": "Mira-Type Long-Period Variable Star",
    },
}

SPHEREX_DETECTOR_ARRAYS = [
    {"id": "SPHEREx-D1", "short": "D1", "min_um": 0.75, "max_um": 1.11, "channels": 17, "r_power": 39, "title": "Array D1 (0.75–1.11 µm)"},
    {"id": "SPHEREx-D2", "short": "D2", "min_um": 1.10, "max_um": 1.65, "channels": 17, "r_power": 41, "title": "Array D2 (1.10–1.65 µm)"},
    {"id": "SPHEREx-D3", "short": "D3", "min_um": 1.63, "max_um": 2.43, "channels": 17, "r_power": 41, "title": "Array D3 (1.63–2.43 µm)"},
    {"id": "SPHEREx-D4", "short": "D4", "min_um": 2.40, "max_um": 3.85, "channels": 17, "r_power": 35, "title": "Array D4 (2.40–3.85 µm)"},
    {"id": "SPHEREx-D5", "short": "D5", "min_um": 3.81, "max_um": 4.43, "channels": 17, "r_power": 112, "title": "Array D5 (3.81–4.43 µm)"},
    {"id": "SPHEREx-D6", "short": "D6", "min_um": 4.41, "max_um": 5.01, "channels": 17, "r_power": 128, "title": "Array D6 (4.41–5.01 µm)"},
]

MOLECULAR_MARKERS = [
    (1.40, "H₂O"),
    (1.90, "H₂O"),
    (2.70, "H₂O / OH"),
    (3.05, "H₂O Ice"),
    (3.30, "CH₄"),
    (4.26, "CO₂"),
    (4.67, "CO"),
]

# =============================================================================
# 2. Bilingual Localization (English / Arabic)
# =============================================================================
TEXT = {
    "en": {
        "subtitle": "SPHEREx 102-Channel Near-Infrared & Multi-Mission Time-Domain Observatory",
        "target_label": "Target Object / Coordinates:",
        "custom_option": "Custom RA / Dec",
        "ra_label": "RA (deg or HMS)",
        "dec_label": "Dec (deg or DMS)",
        "active_coords": "Active ICRS J2000 Coordinates (HMS / DMS):",
        "custom_desc": "Custom sky patch defined by user",
        "fov_label": "Field of View (arcmin):",
        "release_label": "SPHEREx Release:",
        "kpi_coords": "ICRS J2000 Center",
        "kpi_spherex": "SPHEREx Real Frames",
        "kpi_channels": "Spectral Channels Sampled",
        "kpi_pm": "Max Gaia DR3 PM",
        "sec_data": "01. SPHEREx & Multi-Mission Archives",
        "source_label": "Time-Domain Image Feed:",
        "sources": {
            "spherex_only": "SPHEREx Only (DR1 / QR3 True L2)",
            "spherex_plus_wise": "SPHEREx + NEOWISE/unWISE Baseline",
            "wise_w2": "NEOWISE/unWISE W2 (4.6 µm)",
            "wise_w1": "NEOWISE/unWISE W1 (3.4 µm)",
        },
        "zodi_label": "Subtract SPHEREx ZODI Background (IMAGE − ZODI)",
        "mask_flags_label": "Mask Flagged SPHEREx Pixels (FLAGS != 0)",
        "max_spherex_label": "Max SPHEREx Cutouts to Download:",
        "sec_mode": "02. Time-Domain & Spectral Mode",
        "modes": {
            "blink": "Interactive Blink Video (Chronological)",
            "diff": "Difference Imaging (I_B − I_A) & Moving Object Detection",
            "coadd": "Mean / Median Stacking (SNR Boost)",
            "arrays": "6 Detector Arrays vs 102 Spectral Channels",
            "side": "Side-by-Side Chronological Comparison",
            "aladin": "Aladin Lite v3 Interactive Sky",
        },
        "sec_stretch": "03. Display, Alignment & Overlays",
        "stretch_label": "Image Stretch:",
        "stretches": {
            "asinh": "Asinh (Recommended IR)",
            "log": "Logarithmic (Faint Sources)",
            "sqrt": "Square Root",
            "linear": "Linear",
        },
        "invert_label": "Invert Polarity (White Sky)",
        "psf_match_label": "Smooth / Match PSF Before Subtraction",
        "gaia_label": "Overlay Gaia DR3 (Propagated to Frame Epoch)",
        "pm_label": "PM Vector Projection (Years):",
        "sed_label": "Calibrated Spectral Analysis (SPHEREx + Catalogs)",
        "cadc_label": "Query CSA CADC / NEOSSat",
        "query_btn": "🚀 Fetch Live Archive Data",
        "refresh_btn": "🔄 Clear Cache & Refresh Releases",
        "coord_err": "Invalid coordinate format:",
        "loading_title": "Querying NASA/IPAC IRSA (TAP & Cutouts), ESA Gaia DR3 & CSA CADC...",
        "loading_done": "Loaded {n_spherex} SPHEREx + {n_wise} WISE frames, {n_gaia} Gaia DR3, {n_neossat} NEOSSat obs",
        "share_label": "Embed / Share URL Parameters:",
        "no_frames": "No science frames returned for the selected feed/filters. Check diagnostics below or switch archive feed.",
        "timeline_title": "Chronological Observation Timeline",
        "diff_a": "Epoch A (Baseline / Older):",
        "diff_b": "Epoch B (Target / Newer):",
        "diff_unit_warn": "Unit / Band Mismatch Warning: Epochs A and B differ in mission/unit or detector array. Comparing identical detector/band epochs is required for rigorous photometry.",
        "diff_caption": "Red = newer object position / brightening | Blue = older position / dimming • Sky background normalized & WCS-aligned before subtraction.",
        "diff_need_two": "At least two valid frames are required for difference imaging.",
        "moving_title": "Automated Moving Object / Transient Candidates (SNR > 4σ)",
        "moving_none": "No >4σ residual dipole/transient candidates detected in the current pair.",
        "coadd_method": "Stacking Method:",
        "coadd_caption": "Stacking aligned observations increases Signal-to-Noise Ratio (SNR). Median rejects cosmic rays & transient artifacts; Mean maximizes linear sensitivity.",
        "arrays_title": "SPHEREx Architecture: 6 Broadband Detector Arrays (D1–D6) Housing 102 LVF Spectral Channels",
        "arrays_explain": "SPHEREx uses Linear Variable Filters (LVFs): each of the 6 detector arrays spans ~17 narrow spectral channels (102 total across 0.75–5.01 µm). Below are real exposures stacked per array alongside exact WCS-WAVE central wavelengths.",
        "arrays_none": "No SPHEREx exposures matched this array in the current query window.",
        "channel_explorer": "Explore Available Spectral Channels (Sorted by Exact WCS-WAVE λ)",
        "meta_expander": "📋 Full Observation Metadata, FITS Units, Contamination Flags & Diagnostics",
        "diag_expander": "🛠️ Archive Connection Diagnostics & Non-Fatal Warnings",
        "sed_title": "Calibrated Spectral Energy Distribution (SPHEREx MJy/sr→mJy ± σ + IRSA 2MASS/AllWISE)",
        "sed_xlabel": "Wavelength (µm) — SPHEREx Full Spectral Coverage: 0.75–5.01 µm",
        "sed_ylabel": "Calibrated Flux Density (mJy)",
        "sed_note_unwise": "Note: Raw unWISE frames are in Vega-normalized image counts (not MJy/sr); they are excluded from the calibrated mJy axis to preserve photometric accuracy.",
        "sed_empty": "No calibrated SPHEREx mJy aperture points or 2MASS/AllWISE catalog matches available for this coordinate.",
        "gaia_title": "ESA Gaia DR3 Catalog (Ref Epoch J2016.0, Proper Motion & RUWE Astrometric Flag)",
        "cadc_title": "🇨🇦 CSA CADC / NEOSSat Archive Observations (Within 30' Radius)",
        "cadc_empty": "No NEOSSat observations archived within 30 arcmin of this coordinate.",
        "player": {"play": "Play", "pause": "Pause", "prev": "Prev", "next": "Next", "fps": "Speed (FPS)", "crosshair": "Target Crosshair"},
        "target_descs": {
            "Barnard's Star": "Fastest proper-motion star (~10.36\"/yr) — tests Gaia DR3 epoch propagation & IR Blink/Diff.",
            "Epsilon Indi B": "Binary brown dwarf pair (Ba/Bb, ~4.7\"/yr) orbiting Epsilon Indi.",
            "WISE 0855": "Coldest known Y-dwarf (~250 K, ~8.1\"/yr) — invisible in optical, bright at 4.6 µm.",
            "RW Aurigae": "Young variable T Tauri binary — tests IR photometric variability over time.",
            "R Draconis": "Mira-type long-period variable — located in SPHEREx Deep Field (>1,000 DR1 exposures!).",
        },
    },
    "ar": {
        "subtitle": "مرصد التحليل الزمني وطيف SPHEREx ذي الـ 102 قناة تحت الحمراء",
        "target_label": "الهدف الفلكي / الإحداثيات:",
        "custom_option": "إدخال إحداثيات مخصصة (RA / Dec)",
        "ra_label": "المطلع المستقيم RA (درجات أو HMS)",
        "dec_label": "الميل Dec (درجات أو DMS)",
        "active_coords": "إحداثيات ICRS J2000 النشطة (HMS / DMS):",
        "custom_desc": "رقعة سماء مخصصة من اختيار المستخدم",
        "fov_label": "مجال الرؤية FOV (دقيقة قوسية):",
        "release_label": "إصدار بيانات SPHEREx:",
        "kpi_coords": "المركز المرجعي J2000",
        "kpi_spherex": "صور SPHEREx الحقيقية",
        "kpi_channels": "القنوات الطيفية المرصودة",
        "kpi_pm": "أسرع حركة في Gaia DR3",
        "sec_data": "01. أرشيف SPHEREx والبعثات المساعدة",
        "source_label": "مصدر الصور الزمنية:",
        "sources": {
            "spherex_only": "بيانات SPHEREx فقط (DR1 / QR3 الحقيقية)",
            "spherex_plus_wise": "SPHEREx + خط أساس NEOWISE/unWISE",
            "wise_w2": "NEOWISE/unWISE نطاق W2 (4.6 µm)",
            "wise_w1": "NEOWISE/unWISE نطاق W1 (3.4 µm)",
        },
        "zodi_label": "طرح خلفية الغبار البروجي (IMAGE − ZODI)",
        "mask_flags_label": "حجب البكسلات الملوثة (FLAGS != 0)",
        "max_spherex_label": "أقصى عدد لقطات SPHEREx للتحميل:",
        "sec_mode": "02. وضع التحليل الزمني والطيفي",
        "modes": {
            "blink": "فيديو التتابع التفاعلي (Blink زمني)",
            "diff": "طرح الصور (I_B − I_A) وكشف الأجسام المتحركة",
            "coadd": "الجمع (Mean / Median Stacking) لرفع الوضوح",
            "arrays": "المصفوفات الستة (D1–D6) مقابل الـ 102 قناة طيفية",
            "side": "مقارنة زمنية جنباً إلى جنب",
            "aladin": "مستعرض السماء التفاعلي (Aladin Lite v3)",
        },
        "sec_stretch": "03. المعالجة والتراكبات الفلكية",
        "stretch_label": "تمديد التباين (Stretch):",
        "stretches": {
            "asinh": "Asinh (الأنسب للأشعة تحت الحمراء)",
            "log": "لوغاريتمي (للأجسام الخافتة)",
            "sqrt": "الجذر التربيعي",
            "linear": "خطي (Linear)",
        },
        "invert_label": "عكس الألوان (سماء بيضاء)",
        "psf_match_label": "توحيد حدة الصورة (PSF Smoothing) قبل الطرح",
        "gaia_label": "تراكب Gaia DR3 (مُحدَّث لحقبة كل صورة)",
        "pm_label": "طول سهم الحركة (عدد السنوات):",
        "sed_label": "التحليل الطيفي المُعاير (SPHEREx + الفهارس)",
        "cadc_label": "ربط أرشيف وكالة الفضاء الكندية CADC / NEOSSat",
        "query_btn": "🚀 جلب البيانات الحية من الأرشيف",
        "refresh_btn": "🔄 مسح الكاش وفحص أحدث الإطلاقات",
        "coord_err": "صيغة الإحداثيات غير صحيحة:",
        "loading_title": "جاري الاستعلام بالتوازي من NASA/IPAC IRSA و ESA Gaia DR3 و CSA CADC...",
        "loading_done": "تم التحميل: {n_spherex} SPHEREx + {n_wise} WISE، و{n_gaia} نجم Gaia، و{n_neossat} رصد NEOSSat",
        "share_label": "رابط المشاركة والتضمين:",
        "no_frames": "لم يتم إرجاع صور صالحة للمصدر المختار. راجعي سجل التشخيص بالأسفل أو غيري المصدر.",
        "timeline_title": "الخط الزمني الترتيبي للأرصاد (Chronological Timeline)",
        "diff_a": "الحقبة أ (الأساس / الأقدم):",
        "diff_b": "الحقبة ب (الهدف / الأحدث):",
        "diff_unit_warn": "تحذير اختلاف الوحدات أو النطاق: الحقبتان من بعثتين أو مصفوفتين مختلفتين؛ الدقة العلمية تتطلب طرح صور من نفس المصفوفة/النطاق والوحدة.",
        "diff_caption": "🔴 الأحمر = الموقع الأحدث أو زيادة السطوع | 🔵 الأزرق = الموقع القديم أو الخفوت • تم تطبيع الخلفية ومطابقة الـ WCS قبل الطرح.",
        "diff_need_two": "يلزم إطاران صالحان على الأقل لإجراء الطرح الزمني.",
        "moving_title": "المرشحات التلقائية للأجسام المتحركة والمتغيرة (SNR > 4σ)",
        "moving_none": "لم يتم رصد بقع ثنائية القطب أو تغيرات تتجاوز 4σ في هذا الزوج.",
        "coadd_method": "طريقة التجميع:",
        "coadd_caption": "دمج الصور يرفع نسبة الإشارة للضوضاء (SNR). الوسيط (Median) يزيل الأشعة الكونية، والمتوسط (Mean) يعظم الحساسية.",
        "arrays_title": "بنية SPHEREx: 6 مصفوفات كواشف (D1–D6) تحوي 102 قناة طيفية ضيقة (LVF)",
        "arrays_explain": "يستخدم SPHEREx مرشحات متغيرة خطياً (LVF): كل مصفوفة تغطي ~17 قناة طيفية ضيقة (إجمالي 102 قناة بين 0.75 و5.01 µm). هنا تُجمع الأرصاد الحقيقية لكل مصفوفة وتُعرض أطوالها الموجية الفعلية من WCS-WAVE.",
        "arrays_none": "لا توجد أرصاد SPHEREx محملة لهذه المصفوفة في نافذة الاستعلام الحالية.",
        "channel_explorer": "استكشاف القنوات الطيفية المتاحة (مرتبة حسب الطول الموجي الحقيقي WCS-WAVE)",
        "meta_expander": "📋 البيانات الوصفية الكاملة، وحدات FITS، أعلام التلوث (FLAGS)، والتشخيص",
        "diag_expander": "🛠️ سجل تشخيص الاتصال بالأرشيفات والتحذيرات",
        "sed_title": "التوزيع الطيفي المُعاير (تدفق SPHEREx بوحدة mJy ± σ + فهارس 2MASS و AllWISE)",
        "sed_xlabel": "الطول الموجي (µm) — تغطية SPHEREx الكاملة: 0.75–5.01 µm",
        "sed_ylabel": "كثافة التدفق المُعايرة (mJy)",
        "sed_note_unwise": "ملاحظة علمية: صور unWISE الخام بوحدة عدّ معايرة نسبياً وليست MJy/sr، لذلك لا تُدمج في محور الـ mJy حفاظاً على الدقة الفيزيائية.",
        "sed_empty": "لا توجد نقاط تدفق معايرة بـ mJy أو مطابقات في 2MASS/AllWISE لهذا المركز.",
        "gaia_title": "فهرس القمر الأوروبي Gaia DR3 (الحقبة المرجعية J2016.0، الحركة الذاتية، ومعامل الجودة RUWE)",
        "cadc_title": "🇨🇦 أرصاد القمر الصناعي الكندي NEOSSat عبر أرشيف CADC (نطاق 30 دقيقة قوسية)",
        "cadc_empty": "لا توجد أرصاد مسجلة لـ NEOSSat ضمن 30 دقيقة قوسية من هذا الموقع.",
        "player": {"play": "تشغيل", "pause": "إيقاف", "prev": "السابق", "next": "التالي", "fps": "السرعة (FPS)", "crosshair": "علامة المركز"},
        "target_descs": {
            "Barnard's Star": "أسرع نجم حركةً ذاتية (~10.36\"/سنة) — مثالي لاختبار تحديث حقبة Gaia DR3 و الـ Blink/Diff.",
            "Epsilon Indi B": "زوج أقزام بنية (Ba/Bb، ~4.7\"/سنة) يدوران حول نجم قريب.",
            "WISE 0855": "أبرد قزم بني معروف (~250 كلفن، ~8.1\"/سنة) — لا يُرى بالضوء المرئي ويسطع عند 4.6 µm.",
            "RW Aurigae": "نجم ثنائي متغير السطوع (T Tauri) — لاختبار تغير الإضاءة عبر الزمن.",
            "R Draconis": "نجم متغير نابض (Mira) — يقع في حقل SPHEREx العميق (أكثر من 1000 لقطة في DR1!).",
        },
    },
}

# =============================================================================
# 3. Resilient HTTP Networking (Timeouts & Exponential Backoff)
# =============================================================================
def create_http_session(total_retries: int = 3, backoff_factor: float = 0.8) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "SPHEREx-SkyBlink/6.0 (NASA-SpaceApps-Research-Tool)"})
    return session


def fetch_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: dict | None = None,
    timeout: float = 15.0,
    manual_attempts: int = 2,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(manual_attempts):
        try:
            resp = session.request(method=method, url=url, params=params, data=data, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504) and attempt + 1 < manual_attempts:
                time.sleep(1.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < manual_attempts:
                time.sleep(0.8 * (attempt + 1))
    assert last_exc is not None
    raise last_exc

# =============================================================================
# 4. FITS MEF Processor (IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE, Alignment, Diff)
# =============================================================================
def parse_sky_coord(ra_input: str, dec_input: str) -> SkyCoord:
    ra_str, dec_str = ra_input.strip(), dec_input.strip()
    try:
        return SkyCoord(ra=float(ra_str) * u.deg, dec=float(dec_str) * u.deg, frame="icrs")
    except ValueError:
        return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")


def mjd_to_jyear(mjd: float) -> float:
    if not np.isfinite(mjd) or mjd <= 0:
        return 2000.0
    return 2000.0 + (float(mjd) - 51544.5) / 365.25


def _match_detector_id(band_name: str, header_detector=None) -> str:
    if header_detector is not None:
        det_str = str(header_detector).strip()
        if det_str.isdigit():
            return f"SPHEREx-D{det_str}"
        if det_str.upper().startswith("D") and det_str[1:].isdigit():
            return f"SPHEREx-{det_str.upper()}"
    for arr in SPHEREX_DETECTOR_ARRAYS:
        if arr["short"] in band_name.upper():
            return arr["id"]
    return band_name or "SPHEREx"


def _extract_wcs_wave(hdul: fits.HDUList, img_hdr: fits.Header, spatial_wcs: WCS, target_coord: SkyCoord, img_shape: tuple[int, int]):
    if "WCS-WAVE" not in hdul:
        return None, None, "No WCS-WAVE HDU"
    try:
        spectral_wcs = WCS(header=img_hdr, fobj=hdul, key="W")
        spectral_wcs.sip = None
        px, py = spatial_wcs.world_to_pixel(target_coord)
        h, w = img_shape
        if not (np.isfinite(px) and np.isfinite(py)):
            px, py = (w - 1) / 2.0, (h - 1) / 2.0
        px = float(np.clip(px, 0.0, max(w - 1, 0)))
        py = float(np.clip(py, 0.0, max(h - 1, 0)))
        world = spectral_wcs.pixel_to_world(px, py)
        if isinstance(world, (list, tuple)) and len(world) >= 1:
            lam_raw = world[0]
            bw_raw = world[1] if len(world) >= 2 else None
        else:
            lam_raw = world
            bw_raw = None

        def _to_um(val):
            if val is None:
                return None
            if hasattr(val, "to_value"):
                return float(val.to_value(u.um))
            fval = float(val)
            if not np.isfinite(fval):
                return None
            return fval * 1e6 if fval < 1e-3 else fval

        lam_um = _to_um(lam_raw)
        bw_um = _to_um(bw_raw)
        if lam_um is not None and 0.5 <= lam_um <= 6.0:
            return lam_um, bw_um, "WCS-WAVE (WAVE-TAB)"
    except Exception as exc:
        return None, None, f"WCS-WAVE fallback ({exc})"
    return None, None, "WCS-WAVE out of range"


def parse_spherex_mef(
    raw_bytes: bytes,
    target_coord: SkyCoord,
    *,
    subtract_zodi: bool = True,
    mask_flagged: bool = False,
    meta: dict | None = None,
) -> dict | None:
    meta = meta or {}
    with fits.open(io.BytesIO(raw_bytes)) as hdul:
        if "IMAGE" in hdul:
            img_hdu = hdul["IMAGE"]
        elif len(hdul) > 1 and hdul[1].data is not None:
            img_hdu = hdul[1]
        else:
            img_hdu = hdul[0]

        if img_hdu.data is None:
            return None

        img_data = img_hdu.data.astype(np.float64)
        img_hdr = img_hdu.header
        spatial_wcs = WCS(img_hdr).celestial
        bunit = str(img_hdr.get("BUNIT", "MJy/sr")).strip()

        try:
            pixel_area_deg2 = float(proj_plane_pixel_area(spatial_wcs))
            pixel_area_sr = pixel_area_deg2 * ((math.pi / 180.0) ** 2)
        except Exception:
            pixel_area_sr = None

        zodi_median = None
        zodi_subtracted = False
        if "ZODI" in hdul and hdul["ZODI"].data is not None:
            zodi_data = hdul["ZODI"].data.astype(np.float64)
            if zodi_data.shape == img_data.shape:
                finite_z = zodi_data[np.isfinite(zodi_data)]
                if finite_z.size > 0:
                    zodi_median = float(np.median(finite_z))
                if subtract_zodi:
                    img_data = img_data - np.nan_to_num(zodi_data, nan=0.0)
                    zodi_subtracted = True

        variance_data = None
        if "VARIANCE" in hdul and hdul["VARIANCE"].data is not None:
            var_raw = hdul["VARIANCE"].data.astype(np.float64)
            if var_raw.shape == img_data.shape:
                variance_data = np.where(np.isfinite(var_raw) & (var_raw >= 0), var_raw, np.nan)

        flagged_pct = 0.0
        center_flag_val = 0
        if "FLAGS" in hdul and hdul["FLAGS"].data is not None:
            flags_data = hdul["FLAGS"].data
            if flags_data.shape == img_data.shape:
                flagged_pct = 100.0 * float(np.count_nonzero(flags_data)) / max(flags_data.size, 1)
                cy, cx = flags_data.shape[0] // 2, flags_data.shape[1] // 2
                center_flag_val = int(flags_data[cy, cx])
                if mask_flagged:
                    img_data = np.where(flags_data == 0, img_data, np.nan)

        if int(np.isfinite(img_data).sum()) < 16:
            return None

        mjd_obs = float(img_hdr.get("MJD-OBS", meta.get("t_min", 0.0) or 0.0))
        date_obs = str(img_hdr.get("DATE-OBS", ""))
        detector_id = _match_detector_id(str(meta.get("band", "")), img_hdr.get("DETECTOR"))

        lam_um, bw_um, wave_src = _extract_wcs_wave(hdul, img_hdr, spatial_wcs, target_coord, img_data.shape)
        if lam_um is None and meta.get("em_min_um") and meta.get("em_max_um"):
            lam_um = 0.5 * (float(meta["em_min_um"]) + float(meta["em_max_um"]))
            bw_um = float(meta["em_max_um"]) - float(meta["em_min_um"])
            wave_src = "ObsCore Passband Midpoint"

        lam_tag = f"{lam_um:.3f} µm" if lam_um is not None else detector_id
        time_tag = date_obs[:16].replace("T", " ") if date_obs else f"MJD {mjd_obs:.2f}"

        return {
            "obs_id": meta.get("obs_id", f"{detector_id}_{mjd_obs:.5f}"),
            "label": f"{detector_id} • {lam_tag} ({time_tag})",
            "data": img_data.astype(np.float32),
            "variance": variance_data.astype(np.float32) if variance_data is not None else None,
            "wcs": spatial_wcs,
            "pixel_area_sr": pixel_area_sr,
            "bunit": bunit,
            "mjd": mjd_obs,
            "date_obs": date_obs,
            "jyear": mjd_to_jyear(mjd_obs),
            "mission": meta.get("collection", "SPHEREx"),
            "band": detector_id,
            "wavelength_um": lam_um,
            "bandwidth_um": bw_um,
            "wave_source": wave_src,
            "flagged_pct": flagged_pct,
            "center_flag": center_flag_val,
            "zodi_median": zodi_median,
            "zodi_subtracted": zodi_subtracted,
            "quality_flag": meta.get("quality_flag", ""),
            "flags": f"FLAGS!=0: {flagged_pct:.1f}% | CenterFlag: {center_flag_val} | ZodiSub: {'Yes' if zodi_subtracted else 'No'}",
        }


def build_reference_wcs(coord: SkyCoord, fov_arcmin: float, npix: int = 160) -> WCS:
    pixel_scale_deg = (fov_arcmin / 60.0) / npix
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2.0 + 0.5, npix / 2.0 + 0.5]
    w.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.0
    return w


def align_frame_safely(fr: dict, ref_wcs: WCS, shape_out=(160, 160)) -> dict | None:
    try:
        reproj, _ = reproject_interp((fr["data"], fr["wcs"]), ref_wcs, shape_out=shape_out)
        valid_count = int(np.isfinite(reproj).sum())
        if valid_count >= 150 and float(np.nanstd(reproj)) > 0:
            var_reproj = None
            if fr.get("variance") is not None:
                try:
                    vr, _ = reproject_interp((fr["variance"], fr["wcs"]), ref_wcs, shape_out=shape_out)
                    var_reproj = np.where(np.isfinite(vr) & (vr >= 0), vr, np.nan)
                except Exception:
                    var_reproj = None
            bg_med = float(np.nanmedian(reproj))
            return {
                **fr,
                "aligned": reproj.astype(np.float32),
                "aligned_variance": var_reproj,
                "bg_median": bg_med,
                "used_wcs": ref_wcs,
            }
    except Exception:
        pass

    if int(np.isfinite(fr["data"]).sum()) >= 64 and float(np.nanstd(fr["data"])) > 0:
        bg_med = float(np.nanmedian(fr["data"]))
        return {
            **fr,
            "aligned": fr["data"],
            "aligned_variance": fr.get("variance"),
            "bg_median": bg_med,
            "used_wcs": fr["wcs"],
        }
    return None


def apply_stretch(data: np.ndarray, mode_key: str) -> np.ndarray:
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return np.zeros_like(data, dtype=np.float32)
    med = float(np.median(data[finite_mask]))
    clean = np.where(finite_mask, data, med)
    vmin, vmax = np.percentile(clean, [2.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(clean, dtype=np.float32)
    norm = np.clip((clean - vmin) / (vmax - vmin), 0.0, 1.0)
    if mode_key == "asinh":
        return np.arcsinh(10.0 * norm) / np.arcsinh(10.0)
    elif mode_key == "log":
        return np.log10(1.0 + 99.0 * norm) / 2.0
    elif mode_key == "sqrt":
        return np.sqrt(norm)
    return norm


def compute_difference_and_candidates(
    frame_a: dict,
    frame_b: dict,
    *,
    psf_smooth_sigma: float = 0.0,
    snr_threshold: float = 4.0,
) -> dict:
    warnings = []
    u_a = "".join(str(frame_a.get("bunit", "")).lower().split())
    u_b = "".join(str(frame_b.get("bunit", "")).lower().split())
    if u_a != u_b:
        warnings.append(f"Unit mismatch: {frame_a.get('bunit')} vs {frame_b.get('bunit')}")
    if frame_a.get("band") != frame_b.get("band"):
        warnings.append(f"Band/Detector mismatch: {frame_a.get('band')} vs {frame_b.get('band')}")
    lam_a, lam_b = frame_a.get("wavelength_um"), frame_b.get("wavelength_um")
    if lam_a is not None and lam_b is not None and abs(float(lam_b) - float(lam_a)) > 0.35:
        warnings.append(f"Wavelength offset: {lam_a:.3f} µm vs {lam_b:.3f} µm")

    a_arr = np.nan_to_num(frame_a["aligned"], nan=frame_a.get("bg_median", 0.0)) - frame_a.get("bg_median", 0.0)
    b_arr = np.nan_to_num(frame_b["aligned"], nan=frame_b.get("bg_median", 0.0)) - frame_b.get("bg_median", 0.0)

    if psf_smooth_sigma > 0.0:
        a_arr = gaussian_filter(a_arr, sigma=psf_smooth_sigma)
        b_arr = gaussian_filter(b_arr, sigma=psf_smooth_sigma)

    diff = b_arr - a_arr
    med_d = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med_d)))
    noise_sigma = max(1.4826 * mad, float(np.std(diff)) * 0.1, 1e-12)

    var_a = frame_a.get("aligned_variance")
    var_b = frame_b.get("aligned_variance")
    if var_a is not None and var_b is not None and var_a.shape == diff.shape and var_b.shape == diff.shape:
        var_sum = np.nan_to_num(var_a, nan=noise_sigma ** 2) + np.nan_to_num(var_b, nan=noise_sigma ** 2)
        sigma_map = np.maximum(np.sqrt(np.maximum(var_sum, 0.0)), noise_sigma)
    else:
        sigma_map = np.full_like(diff, noise_sigma, dtype=np.float64)

    snr_map = diff / sigma_map
    local_max = (snr_map == maximum_filter(snr_map, size=7)) & (snr_map > snr_threshold)
    local_min = (snr_map == minimum_filter(snr_map, size=7)) & (snr_map < -snr_threshold)
    wcs = frame_a["used_wcs"]
    h, w = diff.shape

    candidates = []
    for polarity, mask in [("Brightened / New Position (+)", local_max), ("Dimmed / Old Position (-)", local_min)]:
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            if 4 <= y < h - 4 and 4 <= x < w - 4:
                sky = wcs.pixel_to_world(float(x), float(y))
                snr = float(snr_map[y, x])
                candidates.append({
                    "type": polarity,
                    "x_px": int(x),
                    "y_px": int(y),
                    "ra_deg": round(float(sky.ra.deg), 6),
                    "dec_deg": round(float(sky.dec.deg), 6),
                    "snr": round(snr, 2),
                })

    candidates.sort(key=lambda c: abs(c["snr"]), reverse=True)
    return {
        "diff": diff,
        "noise_sigma": noise_sigma,
        "warnings": warnings,
        "candidates": candidates[:15],
        "wcs": wcs,
    }


def parse_datalink_cutout_url(xml_text: str, ra_deg: float, dec_deg: float, size_deg: float) -> str | None:
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        access_url = None
        for res in root.findall(".//RESOURCE"):
            if res.attrib.get("ID") == "ibe-cutout":
                for p in res.findall("./PARAM"):
                    if p.attrib.get("name") == "accessURL":
                        access_url = p.attrib.get("value")

        for tr in root.findall(".//TABLEDATA/TR"):
            tds = [td.text or "" for td in tr.findall("TD")]
            if len(tds) >= 13 and tds[4] == "#cutout" and tds[12]:
                base = access_url or "https://irsa.ipac.caltech.edu/ibe/cutout"
                return f"{base}?ra={ra_deg:.6f}&dec={dec_deg:.6f}&size={size_deg:.4f}&path={tds[12]}"
    except Exception:
        return None
    return None

# =============================================================================
# 5. SPHEREx Archive Client (IRSA TAP, ObsCore DataLink, SIAv2, IBE Cutout)
# =============================================================================
def _query_irsa_tap_json(session, adql: str) -> tuple[list[str], list[list]]:
    resp = fetch_with_retry(
        session,
        "POST",
        IRSA_TAP_SYNC_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql},
        timeout=30.0,
    )
    payload = resp.json()
    tbl = payload["VOTABLE"]["RESOURCE_ARRAY"][0]["TABLE"]
    fields = [f["<xmlattr>"]["name"] for f in tbl["FIELD_ARRAY"]]
    rows = tbl["DATA"].get("TABLEDATA", [])
    return fields, rows


def _discover_spherex_rows(
    session,
    ra_deg: float,
    dec_deg: float,
    release_filter: str,
    max_frames: int,
    diagnostics: list[str],
) -> list[dict]:
    rows_out: list[dict] = []
    seen_keys: set[tuple] = set()

    col_clause = ""
    if release_filter != "all":
        safe_col = release_filter.replace("'", "")
        col_clause = f"AND o.collection = '{safe_col}'"

    # 1. Primary Query: spherex.artifact JOIN spherex.plane JOIN spherex.observation
    try:
        adql = (
            f"SELECT TOP 300 "
            f"a.uri, p.planeid, p.energy_bandpassname, p.energy_bounds_lower, p.energy_bounds_upper, "
            f"p.time_bounds_lower, p.quality_flag, o.collection AS obs_collection "
            f"FROM spherex.artifact a "
            f"JOIN spherex.plane p ON a.planeid = p.planeid "
            f"JOIN spherex.observation o ON p.obsid = o.obsid "
            f"WHERE a.producttype = 'science' "
            f"AND 1 = CONTAINS(POINT('ICRS', {ra_deg:.6f}, {dec_deg:.6f}), p.poly) "
            f"{col_clause} "
            f"ORDER BY p.time_bounds_lower ASC"
        )
        fields, raw_rows = _query_irsa_tap_json(session, adql)
        idx = {f: i for i, f in enumerate(fields)}
        for r in raw_rows:
            uri = str(r[idx["uri"]])
            if not uri.lower().endswith(".fits"):
                continue
            planeid = str(r[idx["planeid"]])
            band = str(r[idx["energy_bandpassname"]])
            t_min = float(r[idx["time_bounds_lower"]]) if r[idx["time_bounds_lower"]] else 0.0
            dedup_key = (planeid, band, round(t_min, 5))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            ibe_path = uri[len("ibe/data/"):] if uri.startswith("ibe/data/") else uri
            rows_out.append({
                "obs_id": planeid,
                "ibe_path": ibe_path,
                "band": band,
                "em_min_um": float(r[idx["energy_bounds_lower"]]) * 1e6 if r[idx["energy_bounds_lower"]] else None,
                "em_max_um": float(r[idx["energy_bounds_upper"]]) * 1e6 if r[idx["energy_bounds_upper"]] else None,
                "t_min": t_min,
                "quality_flag": str(r[idx["quality_flag"]] or ""),
                "collection": str(r[idx["obs_collection"]]),
            })
        if rows_out:
            diagnostics.append(f"SPHEREx TAP (artifact+plane+obs): found {len(rows_out)} unique science frames.")
    except Exception as exc:
        diagnostics.append(f"SPHEREx TAP artifact+plane query warning: {exc}")

    # 2. Secondary Query: spherex.obscore + DataLink
    if not rows_out:
        try:
            obs_col_clause = f"AND obs_collection = '{release_filter.replace(chr(39), '')}'" if release_filter != "all" else ""
            adql_obs = (
                f"SELECT TOP 60 obs_id, obs_collection, energy_bandpassname, em_min, em_max, t_min, access_url "
                f"FROM spherex.obscore "
                f"WHERE dataproduct_type = 'image' "
                f"AND 1 = CONTAINS(POINT('ICRS', {ra_deg:.6f}, {dec_deg:.6f}), poly) "
                f"{obs_col_clause} "
                f"ORDER BY t_min ASC"
            )
            fields, raw_rows = _query_irsa_tap_json(session, adql_obs)
            idx = {f: i for i, f in enumerate(fields)}
            for r in raw_rows:
                obs_id = str(r[idx["obs_id"]])
                band = str(r[idx["energy_bandpassname"]])
                t_min = float(r[idx["t_min"]]) if r[idx["t_min"]] else 0.0
                dedup_key = (obs_id, band, round(t_min, 5))
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                rows_out.append({
                    "obs_id": obs_id,
                    "datalink_url": str(r[idx["access_url"]]),
                    "band": band,
                    "em_min_um": float(r[idx["em_min"]]) * 1e6 if r[idx["em_min"]] else None,
                    "em_max_um": float(r[idx["em_max"]]) * 1e6 if r[idx["em_max"]] else None,
                    "t_min": t_min,
                    "quality_flag": "",
                    "collection": str(r[idx["obs_collection"]]),
                })
            if rows_out:
                diagnostics.append(f"SPHEREx TAP (obscore): found {len(rows_out)} unique frames.")
        except Exception as exc:
            diagnostics.append(f"SPHEREx obscore query warning: {exc}")

    # 3. Tertiary Fallback: IRSA SIAv2 query + VOTable parsing
    if not rows_out:
        try:
            col_param = release_filter if release_filter != "all" else "spherex_dr1"
            resp = fetch_with_retry(
                session,
                "GET",
                IRSA_SIA_URL,
                params={"COLLECTION": col_param, "POS": f"circle {ra_deg:.5f} {dec_deg:.5f} 0.02"},
                timeout=15.0,
            )
            root = ET.fromstring(resp.text)
            for elem in root.iter():
                if "}" in elem.tag:
                    elem.tag = elem.tag.split("}", 1)[1]
            sia_fields = [f.attrib.get("name", "") for f in root.findall(".//TABLE/FIELD")]
            f_idx = {name: i for i, name in enumerate(sia_fields)}
            for tr in root.findall(".//TABLEDATA/TR"):
                tds = [td.text or "" for td in tr.findall("TD")]
                if "access_url" not in f_idx or len(tds) <= f_idx["access_url"]:
                    continue
                obs_id = tds[f_idx["obs_id"]] if "obs_id" in f_idx else f"sia_{len(rows_out)}"
                band = tds[f_idx["energy_bandpassname"]] if "energy_bandpassname" in f_idx else "SPHEREx"
                t_min = float(tds[f_idx["t_min"]]) if ("t_min" in f_idx and tds[f_idx["t_min"]]) else 0.0
                dedup_key = (obs_id, band, round(t_min, 5))
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                rows_out.append({
                    "obs_id": obs_id,
                    "datalink_url": tds[f_idx["access_url"]],
                    "band": band,
                    "em_min_um": float(tds[f_idx["em_min"]]) * 1e6 if ("em_min" in f_idx and tds[f_idx["em_min"]]) else None,
                    "em_max_um": float(tds[f_idx["em_max"]]) * 1e6 if ("em_max" in f_idx and tds[f_idx["em_max"]]) else None,
                    "t_min": t_min,
                    "quality_flag": "",
                    "collection": col_param,
                })
            diagnostics.append(f"SPHEREx SIAv2 ({col_param}): HTTP {resp.status_code}, parsed {len(rows_out)} rows.")
        except Exception as exc:
            diagnostics.append(f"SPHEREx SIAv2 fallback error: {exc}")

    if len(rows_out) <= max_frames:
        return rows_out

    by_band: dict[str, list[dict]] = {}
    for row in rows_out:
        by_band.setdefault(row["band"], []).append(row)

    selected: list[dict] = []
    for band_rows in by_band.values():
        selected.append(band_rows[0])
        if len(band_rows) > 1 and len(selected) < max_frames:
            selected.append(band_rows[-1])
    seen_ids = {id(x) for x in selected}
    for row in rows_out:
        if len(selected) >= max_frames:
            break
        if id(row) not in seen_ids:
            selected.append(row)
            seen_ids.add(id(row))

    selected.sort(key=lambda r: r["t_min"])
    return selected[:max_frames]


def fetch_spherex_cutouts(
    target_coord: SkyCoord,
    fov_arcmin: float,
    *,
    release_filter: str = "all",
    max_frames: int = 8,
    subtract_zodi: bool = True,
    mask_flagged: bool = False,
) -> tuple[list[dict], list[str]]:
    ra_deg = float(target_coord.ra.deg)
    dec_deg = float(target_coord.dec.deg)
    size_deg = max(fov_arcmin / 60.0, 0.03)
    diagnostics: list[str] = []
    session = create_http_session(total_retries=3, backoff_factor=0.9)

    meta_rows = _discover_spherex_rows(session, ra_deg, dec_deg, release_filter, max_frames, diagnostics)
    if not meta_rows:
        return [], diagnostics

    def _download_one(row: dict) -> dict | None:
        cut_url = None
        if row.get("ibe_path"):
            cut_url = f"{IRSA_IBE_CUTOUT_URL}?{urllib.parse.urlencode({'ra': f'{ra_deg:.6f}', 'dec': f'{dec_deg:.6f}', 'size': f'{size_deg:.4f}', 'path': row['ibe_path']})}"
        elif row.get("datalink_url"):
            try:
                dl_resp = fetch_with_retry(session, "GET", row["datalink_url"], timeout=12.0)
                cut_url = parse_datalink_cutout_url(dl_resp.text, ra_deg, dec_deg, size_deg)
            except Exception as exc:
                diagnostics.append(f"DataLink resolve failed for {row['obs_id']}: {exc}")
                return None

        if not cut_url:
            return None

        try:
            resp = fetch_with_retry(session, "GET", cut_url, timeout=18.0, manual_attempts=2)
            return parse_spherex_mef(
                resp.content,
                target_coord,
                subtract_zodi=subtract_zodi,
                mask_flagged=mask_flagged,
                meta=row,
            )
        except Exception as exc:
            diagnostics.append(f"SPHEREx cutout download failed ({row['band']}): {exc}")
            return None

    frames: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_download_one, r) for r in meta_rows]
        for fut in futures:
            try:
                res = fut.result(timeout=30.0)
                if res is not None:
                    frames.append(res)
            except Exception as exc:
                diagnostics.append(f"SPHEREx worker timeout/error: {exc}")

    frames.sort(key=lambda f: f["mjd"])
    return frames, diagnostics

# =============================================================================
# 6. Multi-Mission Archives (Gaia DR3 + Epoch Propagation, unWISE, 2MASS/AllWISE, NEOSSat)
# =============================================================================
def fetch_gaia_dr3_catalog(ra_deg: float, dec_deg: float, fov_arcmin: float) -> tuple[pd.DataFrame, list[str]]:
    diagnostics: list[str] = []
    session = create_http_session(total_retries=2, backoff_factor=0.7)
    radius_deg = max(fov_arcmin / 2.0, 4.5) / 60.0
    adql = (
        f"SELECT TOP 100 source_id, ra, dec, ref_epoch, phot_g_mean_mag, pmra, pmdec, parallax, ruwe "
        f"FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.5f})) "
        f"ORDER BY phot_g_mean_mag ASC"
    )
    try:
        resp = fetch_with_retry(
            session,
            "POST",
            GAIA_TAP_SYNC_URL,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql},
            timeout=15.0,
        )
        payload = resp.json()
        cols = [m["name"] for m in payload.get("metadata", [])]
        rows = payload.get("data", [])
        if not rows:
            diagnostics.append("Gaia DR3 returned 0 sources in search cone.")
            return pd.DataFrame(), diagnostics
        return pd.DataFrame(rows, columns=cols), diagnostics
    except Exception as exc:
        diagnostics.append(f"Gaia DR3 TAP error: {exc}")
        return pd.DataFrame(), diagnostics


def propagate_gaia_to_epoch(gaia_df: pd.DataFrame, target_jyear: float) -> pd.DataFrame:
    if gaia_df is None or len(gaia_df) == 0:
        return pd.DataFrame()
    out = gaia_df.copy()
    ref_ep = pd.to_numeric(out.get("ref_epoch", 2016.0), errors="coerce").fillna(2016.0)
    dt_yr = target_jyear - ref_ep
    ra0 = pd.to_numeric(out["ra"], errors="coerce")
    dec0 = pd.to_numeric(out["dec"], errors="coerce")
    pmra = pd.to_numeric(out.get("pmra", 0.0), errors="coerce").fillna(0.0)
    pmdec = pd.to_numeric(out.get("pmdec", 0.0), errors="coerce").fillna(0.0)

    cos_dec = np.cos(np.deg2rad(dec0)).clip(lower=1e-4)
    out["ra_epoch"] = ra0 + (pmra * dt_yr) / (3_600_000.0 * cos_dec)
    out["dec_epoch"] = dec0 + (pmdec * dt_yr) / 3_600_000.0
    out["pm_total_mas_yr"] = np.hypot(pmra, pmdec)
    return out


def fetch_unwise_epochs(ra_deg: float, dec_deg: float, fov_arcmin: float, band: str = "1") -> tuple[list[dict], list[str]]:
    diagnostics: list[str] = []
    session = create_http_session(total_retries=2, backoff_factor=0.7)
    size_px = int(np.clip((fov_arcmin * 60.0) / 2.75, 48, 220))
    versions = [
        ("allwise", "AllWISE (2010)", 55350.0),
        ("neo1", "NEOWISE-R1 (2014)", 56800.0),
        ("neo3", "NEOWISE-R3 (2016)", 57500.0),
        ("neo5", "NEOWISE-R5 (2018)", 58250.0),
        ("neo7", "NEOWISE-R7 (2020)", 59000.0),
    ]

    def _one(ver: str, label: str, mjd: float) -> dict | None:
        params = {
            "version": ver,
            "ra": f"{ra_deg:.6f}",
            "dec": f"{dec_deg:.6f}",
            "size": str(size_px),
            "bands": band,
            "file_img_m": "on",
        }
        try:
            resp = fetch_with_retry(session, "GET", UNWISE_CUTOUT_URL, params=params, timeout=10.0)
            best_frame = None
            best_score = -1
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".fits") and "-img-" in member.name:
                        fobj = tar.extractfile(member)
                        if fobj is None:
                            continue
                        with fits.open(io.BytesIO(fobj.read())) as hdul:
                            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                            data = hdu.data.astype(np.float32)
                            finite_cnt = int(np.isfinite(data).sum())
                            if finite_cnt > best_score and float(np.nanstd(data)) > 0:
                                best_score = finite_cnt
                                best_frame = {
                                    "obs_id": f"unwise_{ver}_w{band}",
                                    "label": f"{label} • W{band}",
                                    "data": data,
                                    "variance": None,
                                    "wcs": WCS(hdu.header).celestial,
                                    "pixel_area_sr": (2.75 * math.pi / (180.0 * 3600.0)) ** 2,
                                    "bunit": "unWISE Vega DN",
                                    "mjd": mjd,
                                    "date_obs": f"~{int(mjd_to_jyear(mjd))}",
                                    "jyear": mjd_to_jyear(mjd),
                                    "mission": f"WISE/NEOWISE ({ver})",
                                    "band": f"W{band}",
                                    "wavelength_um": 3.35 if band == "1" else 4.60,
                                    "bandwidth_um": 0.66 if band == "1" else 1.04,
                                    "wave_source": "WISE Filter Bandpass",
                                    "flagged_pct": 0.0,
                                    "zodi_median": None,
                                    "zodi_subtracted": False,
                                    "flags": f"Scale: 2.75\"/px | Band W{band} | Unit: unWISE Vega DN",
                                }
            return best_frame
        except Exception as exc:
            diagnostics.append(f"unWISE {ver} W{band} error: {exc}")
            return None

    frames: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_one, v, lbl, mjd) for v, lbl, mjd in versions]
        for fut in futs:
            try:
                res = fut.result(timeout=14.0)
                if res is not None:
                    frames.append(res)
            except Exception as exc:
                diagnostics.append(f"unWISE worker timeout: {exc}")

    frames.sort(key=lambda x: x["mjd"])
    return frames, diagnostics


def _query_irsa_sed_table(session, adql: str) -> tuple[list[str], list[list]]:
    resp = fetch_with_retry(
        session,
        "POST",
        IRSA_TAP_SYNC_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql},
        timeout=15.0,
    )
    tbl = resp.json()["VOTABLE"]["RESOURCE_ARRAY"][0]["TABLE"]
    fields = [f["<xmlattr>"]["name"] for f in tbl["FIELD_ARRAY"]]
    rows = tbl["DATA"].get("TABLEDATA", [])
    return fields, rows


def fetch_irsa_catalog_sed(ra_deg: float, dec_deg: float) -> tuple[pd.DataFrame, list[str]]:
    diagnostics: list[str] = []
    session = create_http_session(total_retries=2, backoff_factor=0.7)
    sed_points: list[dict] = []
    rad_deg = 8.0 / 3600.0

    def _safe_float(val):
        try:
            f = float(val)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    try:
        adql_2m = (
            f"SELECT TOP 1 j_m, j_msigcom, h_m, h_msigcom, k_m, k_msigcom "
            f"FROM fp_psc WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {rad_deg:.6f}))"
        )
        fields, rows = _query_irsa_sed_table(session, adql_2m)
        if rows:
            idx = {f: i for i, f in enumerate(fields)}
            row = rows[0]
            for col, err_col, lam, zp in [
                ("j_m", "j_msigcom", 1.235, 1594.0),
                ("h_m", "h_msigcom", 1.662, 1024.0),
                ("k_m", "k_msigcom", 2.159, 666.7),
            ]:
                mag = _safe_float(row[idx[col]]) if col in idx else None
                if mag is not None:
                    flux_mjy = zp * (10 ** (-0.4 * mag)) * 1000.0
                    err_mag = _safe_float(row[idx[err_col]]) if err_col in idx else None
                    sigma_mjy = flux_mjy * (math.log(10) / 2.5) * err_mag if err_mag is not None else None
                    sed_points.append({
                        "wavelength_um": lam,
                        "flux_mjy": flux_mjy,
                        "sigma_mjy": sigma_mjy,
                        "band": f"2MASS {col[0].upper()}",
                        "source": "2MASS PSC (Vega ZP)",
                    })
    except Exception as exc:
        diagnostics.append(f"2MASS TAP query warning: {exc}")

    try:
        adql_aw = (
            f"SELECT TOP 1 w1mpro, w1sigmpro, w2mpro, w2sigmpro, w3mpro, w3sigmpro "
            f"FROM allwise_p3as_psd WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {rad_deg:.6f}))"
        )
        fields, rows = _query_irsa_sed_table(session, adql_aw)
        if rows:
            idx = {f: i for i, f in enumerate(fields)}
            row = rows[0]
            for col, err_col, lam, zp in [
                ("w1mpro", "w1sigmpro", 3.35, 309.54),
                ("w2mpro", "w2sigmpro", 4.60, 171.787),
                ("w3mpro", "w3sigmpro", 11.56, 31.674),
            ]:
                mag = _safe_float(row[idx[col]]) if col in idx else None
                if mag is not None:
                    flux_mjy = zp * (10 ** (-0.4 * mag)) * 1000.0
                    err_mag = _safe_float(row[idx[err_col]]) if err_col in idx else None
                    sigma_mjy = flux_mjy * (math.log(10) / 2.5) * err_mag if err_mag is not None else None
                    sed_points.append({
                        "wavelength_um": lam,
                        "flux_mjy": flux_mjy,
                        "sigma_mjy": sigma_mjy,
                        "band": col[:2].upper(),
                        "source": "AllWISE PSC (Vega ZP)",
                    })
    except Exception as exc:
        diagnostics.append(f"AllWISE TAP query warning: {exc}")

    return pd.DataFrame(sed_points), diagnostics


def fetch_cadc_neossat(ra_deg: float, dec_deg: float) -> tuple[pd.DataFrame, list[str]]:
    diagnostics: list[str] = []
    session = create_http_session(total_retries=2, backoff_factor=0.7)
    adql = (
        f"SELECT TOP 15 o.observationID, o.target_name, p.time_bounds_lower, p.time_exposure, o.instrument_name "
        f"FROM caom2.Observation o JOIN caom2.Plane p ON o.obsID = p.obsID "
        f"WHERE o.collection = 'NEOSSAT' "
        f"AND 1 = CONTAINS(CIRCLE('ICRS', {ra_deg:.5f}, {dec_deg:.5f}, 0.5), p.position_bounds)"
    )
    try:
        resp = fetch_with_retry(
            session,
            "POST",
            CADC_TAP_SYNC_URL,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
            timeout=14.0,
        )
        return pd.read_csv(io.StringIO(resp.text)), diagnostics
    except Exception as exc:
        diagnostics.append(f"CSA CADC NEOSSat TAP error: {exc}")
        return pd.DataFrame(), diagnostics

# =============================================================================
# 7. Calibrated Spectral Extraction & 6-Array / 102-Channel Summary
# =============================================================================
def extract_calibrated_spherex_spectrum(frames: list[dict], target_coord: SkyCoord) -> pd.DataFrame:
    points: list[dict] = []
    for fr in frames:
        bunit_norm = "".join(str(fr.get("bunit", "")).lower().split())
        lam = fr.get("wavelength_um")
        omega_sr = fr.get("pixel_area_sr")
        if "mjy/sr" not in bunit_norm or lam is None or omega_sr is None or omega_sr <= 0:
            continue

        img = fr["data"]
        wcs = fr["wcs"]
        var = fr.get("variance")
        h, w = img.shape
        try:
            px, py = wcs.world_to_pixel(target_coord)
            if not (np.isfinite(px) and np.isfinite(py)) or px < 2 or px >= w - 2 or py < 2 or py >= h - 2:
                continue
        except Exception:
            continue

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        ap_mask = dist <= 2.5
        sky_mask = (dist >= 5.0) & (dist <= 10.0)

        ap_vals = img[ap_mask]
        sky_vals = img[sky_mask]
        ap_valid = ap_vals[np.isfinite(ap_vals)]
        sky_valid = sky_vals[np.isfinite(sky_vals)]
        if ap_valid.size < 3 or sky_valid.size < 8:
            continue

        sky_bg = float(np.median(sky_valid))
        net_mjy_sr_sum = float(np.sum(ap_valid - sky_bg))
        flux_mjy = net_mjy_sr_sum * float(omega_sr) * 1e9

        sigma_mjy = None
        if var is not None:
            ap_var = var[ap_mask]
            ap_var_valid = ap_var[np.isfinite(ap_var) & (ap_var >= 0)]
            if ap_var_valid.size > 0:
                sky_mad = 1.4826 * float(np.median(np.abs(sky_valid - sky_bg)))
                var_sum = float(np.sum(ap_var_valid)) + ap_valid.size * (sky_mad ** 2) / max(sky_valid.size, 1)
                sigma_mjy = math.sqrt(max(var_sum, 0.0)) * float(omega_sr) * 1e9
        if sigma_mjy is None:
            sky_mad = 1.4826 * float(np.median(np.abs(sky_valid - sky_bg)))
            sigma_mjy = math.sqrt(ap_valid.size) * sky_mad * float(omega_sr) * 1e9

        points.append({
            "obs_id": fr.get("obs_id", ""),
            "band": fr.get("band", "SPHEREx"),
            "wavelength_um": float(lam),
            "bandwidth_um": float(fr["bandwidth_um"]) if fr.get("bandwidth_um") else None,
            "flux_mjy": flux_mjy,
            "sigma_mjy": sigma_mjy,
            "wave_source": fr.get("wave_source", "WCS-WAVE"),
            "mjd": fr.get("mjd", 0.0),
            "source": "SPHEREx L2 Aperture (MJy/sr→mJy)",
        })

    if not points:
        return pd.DataFrame()
    return pd.DataFrame(points).sort_values("wavelength_um")


def build_channel_summary_table(spherex_frames: list[dict]) -> pd.DataFrame:
    rows = []
    for arr in SPHEREX_DETECTOR_ARRAYS:
        matched = [f for f in spherex_frames if f.get("band") == arr["id"]]
        waves = sorted({round(f["wavelength_um"], 3) for f in matched if f.get("wavelength_um") is not None})
        rows.append({
            "Array": arr["id"],
            "Wavelength Range (µm)": f"{arr['min_um']:.2f}–{arr['max_um']:.2f}",
            "Total LVF Channels (Design)": arr["channels"],
            "Resolving Power (R)": f"~{arr['r_power']}",
            "Loaded Frames": len(matched),
            "Sampled λ (µm) [WCS-WAVE]": ", ".join(f"{w:.3f}" for w in waves) if waves else "—",
        })
    return pd.DataFrame(rows)

# =============================================================================
# 8. Rendering & Interactive HTML5 Blink Comparator
# =============================================================================
def render_frame_to_base64(
    img_2d: np.ndarray,
    label: str,
    cmap: str,
    gaia_df: pd.DataFrame | None,
    wcs: WCS,
    show_gaia_overlay: bool,
    *,
    frame_jyear: float = 2020.0,
    pm_years: float = 10.0,
    candidates: list[dict] | None = None,
) -> str:
    fig = plt.figure(figsize=(5.1, 5.1), dpi=120, facecolor="#050814")
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor("#050814")
    ax.imshow(img_2d, origin="lower", cmap=cmap)

    if show_gaia_overlay and gaia_df is not None and len(gaia_df) > 0:
        prop = propagate_gaia_to_epoch(gaia_df, frame_jyear)
        g_coords = SkyCoord(ra=prop["ra_epoch"].values * u.deg, dec=prop["dec_epoch"].values * u.deg, frame="icrs")
        px, py = wcs.world_to_pixel(g_coords)
        valid = (
            np.isfinite(px)
            & np.isfinite(py)
            & (px >= 0)
            & (px < img_2d.shape[1])
            & (py >= 0)
            & (py < img_2d.shape[0])
        )
        ax.scatter(px[valid], py[valid], s=68, edgecolor="#38bdf8", facecolor="none", linewidths=1.2)

        if "pmra" in prop.columns and "pmdec" in prop.columns:
            future = propagate_gaia_to_epoch(prop, frame_jyear + pm_years)
            f_coords = SkyCoord(ra=future["ra_epoch"].values * u.deg, dec=future["dec_epoch"].values * u.deg, frame="icrs")
            px_fut, py_fut = wcs.world_to_pixel(f_coords)
            dx = np.nan_to_num(px_fut[valid] - px[valid], nan=0.0)
            dy = np.nan_to_num(py_fut[valid] - py[valid], nan=0.0)
            ax.quiver(px[valid], py[valid], dx, dy, angles="xy", scale_units="xy", scale=1, color="#facc15", width=0.005)

    if candidates:
        cx = [c["x_px"] for c in candidates]
        cy = [c["y_px"] for c in candidates]
        ax.scatter(cx, cy, s=110, edgecolor="#22c55e", facecolor="none", linewidths=1.4, linestyle="--")

    ax.set_title(label, color="#38bdf8", fontsize=9.2, pad=8)
    ax.coords[0].set_axislabel("RA (ICRS)", color="#94a3b8", fontsize=8)
    ax.coords[1].set_axislabel("Dec (ICRS)", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.grid(color="#1e293b", ls=":", alpha=0.65)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#050814")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_html_blink_player(encoded_frames: list[dict], player_txt: dict, default_fps: float = 2.0):
    frames_json = json.dumps([
        {
            "label": f["label"],
            "meta": f.get("meta", ""),
            "src": f"data:image/png;base64,{f['b64']}",
        }
        for f in encoded_frames
    ])
    play_lbl = json.dumps(player_txt["play"])
    pause_lbl = json.dumps(player_txt["pause"])
    html_code = f"""
    <div style="background:linear-gradient(145deg, rgba(11,18,36,0.95), rgba(5,8,20,0.98)); border:1px solid rgba(56,189,248,0.28); border-radius:12px; padding:14px; color:#e2e8f0; font-family:'Consolas',monospace; box-shadow:0 0 24px rgba(2,132,199,0.12);">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; gap:8px; flex-wrap:wrap;">
        <span id="frameLabel" style="color:#38bdf8; font-weight:bold; font-size:13px;">Loading...</span>
        <span id="frameCounter" style="background:rgba(56,189,248,0.12); color:#7dd3fc; padding:2px 8px; border-radius:4px; font-size:12px;">1 / {len(encoded_frames)}</span>
      </div>
      <div id="frameMeta" style="color:#94a3b8; font-size:11px; margin-bottom:8px;"></div>
      <div style="position:relative; text-align:center; background:#030612; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.06);">
        <img id="blinkImg" style="max-width:100%; max-height:460px; display:block; margin:auto;" />
        <div id="crosshair" style="display:none; position:absolute; top:50%; left:50%; width:26px; height:26px; transform:translate(-50%,-50%); border:1.5px solid #38bdf8; border-radius:50%; box-shadow:0 0 10px #38bdf8; pointer-events:none;"></div>
      </div>
      <div style="display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap;">
        <button onclick="togglePlay()" id="playBtn" style="background:linear-gradient(90deg,#0284c7,#2563eb); color:white; border:none; padding:6px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">{player_txt['pause']}</button>
        <button onclick="stepFrame(-1)" style="background:#0f172a; color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:6px 12px; border-radius:6px; cursor:pointer;">{player_txt['prev']}</button>
        <button onclick="stepFrame(1)" style="background:#0f172a; color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:6px 12px; border-radius:6px; cursor:pointer;">{player_txt['next']}</button>
        <label style="font-size:12px; color:#94a3b8;">{player_txt['fps']}: <span id="fpsVal" style="color:#38bdf8;">{default_fps}</span>
          <input type="range" min="0.5" max="6" step="0.5" value="{default_fps}" oninput="updateFps(this.value)" style="vertical-align:middle;" />
        </label>
        <label style="font-size:12px; color:#94a3b8; cursor:pointer;">
          <input type="checkbox" onchange="document.getElementById('crosshair').style.display = this.checked ? 'block' : 'none'" /> {player_txt['crosshair']}
        </label>
      </div>
      <input type="range" id="scrubber" min="0" max="{max(len(encoded_frames)-1, 0)}" value="0" oninput="jumpFrame(this.value)" style="width:100%; margin-top:10px;" />
    </div>
    <script>
      const frames = {frames_json};
      const PLAY_LBL = {play_lbl};
      const PAUSE_LBL = {pause_lbl};
      let idx = 0, playing = true, fps = {default_fps}, timer = null;
      function showFrame(i) {{
        if (!frames.length) return;
        idx = (i + frames.length) % frames.length;
        document.getElementById('blinkImg').src = frames[idx].src;
        document.getElementById('frameLabel').innerText = frames[idx].label;
        document.getElementById('frameMeta').innerText = frames[idx].meta || '';
        document.getElementById('frameCounter').innerText = (idx + 1) + ' / ' + frames.length;
        document.getElementById('scrubber').value = idx;
      }}
      function startTimer() {{
        if (timer) clearInterval(timer);
        if (playing && frames.length > 1) timer = setInterval(() => showFrame(idx + 1), 1000 / fps);
      }}
      function togglePlay() {{
        playing = !playing;
        document.getElementById('playBtn').innerText = playing ? PAUSE_LBL : PLAY_LBL;
        startTimer();
      }}
      function stepFrame(d) {{ playing = false; document.getElementById('playBtn').innerText = PLAY_LBL; startTimer(); showFrame(idx + d); }}
      function jumpFrame(v) {{ playing = false; document.getElementById('playBtn').innerText = PLAY_LBL; startTimer(); showFrame(parseInt(v)); }}
      function updateFps(v) {{ fps = parseFloat(v); document.getElementById('fpsVal').innerText = fps; startTimer(); }}
      showFrame(0); startTimer();
    </script>
    """
    components.html(html_code, height=625)

# =============================================================================
# 9. Thread-Safe Cached Archive Bundle & Main Streamlit UI
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def cached_archive_bundle(
    ra_deg: float,
    dec_deg: float,
    fov_arcmin: float,
    release_filter: str,
    max_frames: int,
    subtract_zodi: bool,
    mask_flagged: bool,
    want_spherex: bool,
    want_wise: bool,
    wise_band: str,
    want_sed: bool,
    want_cadc: bool,
) -> dict:
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    diagnostics: list[str] = []

    with ThreadPoolExecutor(max_workers=5) as ex:
        fut_spx = ex.submit(
            fetch_spherex_cutouts,
            coord,
            fov_arcmin,
            release_filter=release_filter,
            max_frames=max_frames,
            subtract_zodi=subtract_zodi,
            mask_flagged=mask_flagged,
        ) if want_spherex else None
        fut_wise = ex.submit(fetch_unwise_epochs, ra_deg, dec_deg, fov_arcmin, band=wise_band) if want_wise else None
        fut_gaia = ex.submit(fetch_gaia_dr3_catalog, ra_deg, dec_deg, fov_arcmin)
        fut_sed = ex.submit(fetch_irsa_catalog_sed, ra_deg, dec_deg) if want_sed else None
        fut_cadc = ex.submit(fetch_cadc_neossat, ra_deg, dec_deg) if want_cadc else None

        spherex_frames, wise_frames = [], []
        if fut_spx:
            try:
                spherex_frames, d_spx = fut_spx.result(timeout=45)
                diagnostics.extend(d_spx)
            except Exception as exc:
                diagnostics.append(f"SPHEREx fetch error: {exc}")

        if fut_wise:
            try:
                wise_frames, d_wise = fut_wise.result(timeout=25)
                diagnostics.extend(d_wise)
            except Exception as exc:
                diagnostics.append(f"unWISE fetch error: {exc}")

        try:
            gaia_df, d_gaia = fut_gaia.result(timeout=20)
            diagnostics.extend(d_gaia)
        except Exception as exc:
            gaia_df = pd.DataFrame()
            diagnostics.append(f"Gaia DR3 fetch error: {exc}")

        try:
            sed_df, d_sed = fut_sed.result(timeout=18) if fut_sed else (pd.DataFrame(), [])
            diagnostics.extend(d_sed)
        except Exception as exc:
            sed_df = pd.DataFrame()
            diagnostics.append(f"IRSA catalog SED error: {exc}")

        try:
            cadc_df, d_cadc = fut_cadc.result(timeout=18) if fut_cadc else (pd.DataFrame(), [])
            diagnostics.extend(d_cadc)
        except Exception as exc:
            cadc_df = pd.DataFrame()
            diagnostics.append(f"CADC NEOSSat error: {exc}")

    return {
        "spherex_frames": spherex_frames,
        "wise_frames": wise_frames,
        "gaia_df": gaia_df,
        "sed_df": sed_df,
        "cadc_df": cadc_df,
        "diagnostics": diagnostics,
    }


qp = st.query_params
init_lang_idx = 1 if qp.get("lang", "en") == "ar" else 0

top_col1, top_col2, top_col3 = st.columns([2.8, 1.2, 1.0])
with top_col3:
    lang_choice = st.radio("🌐 Language / اللغة", ["English", "العربية"], index=init_lang_idx, horizontal=True)
lang = "ar" if lang_choice == "العربية" else "en"
t = TEXT[lang]

rtl_css = """
    .stApp { direction: rtl; text-align: right; }
    input, code, pre, .kpi-val { direction: ltr; unicode-bidi: embed; }
""" if lang == "ar" else ""

st.markdown(f"""
<style>
.stApp {{
    background-color: #030612;
    background-image:
        radial-gradient(white, rgba(255,255,255,0.18) 1px, transparent 2px),
        radial-gradient(rgba(56,189,248,0.6), rgba(56,189,248,0.1) 1px, transparent 2px),
        radial-gradient(circle at 18% 12%, rgba(30, 58, 138, 0.28), transparent 45%),
        radial-gradient(circle at 82% 85%, rgba(14, 116, 144, 0.20), transparent 50%);
    background-size: 290px 290px, 190px 190px, 100% 100%, 100% 100%;
    background-position: 0 0, 75px 110px, 0 0, 0 0;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Consolas', sans-serif;
}}
.hud-banner {{
    background: linear-gradient(90deg, rgba(15,23,42,0.88), rgba(14,116,144,0.22));
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
}}
.kpi-card {{
    background: rgba(11, 18, 36, 0.78);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 10px;
    padding: 10px 14px;
    backdrop-filter: blur(6px);
}}
.kpi-lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px; }}
.kpi-val {{ font-size: 15px; font-weight: 700; color: #38bdf8; margin-top: 2px; font-family: 'Consolas', monospace; }}
h1, h2, h3, h4 {{ color: #38bdf8 !important; letter-spacing: 0.4px; }}
div.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    box-shadow: 0 0 16px rgba(2, 132, 199, 0.35);
}}
{rtl_css}
</style>
""", unsafe_allow_html=True)

with top_col1:
    st.markdown(f"""
    <div class="hud-banner">
      <div style="font-size:24px; font-weight:800; color:#38bdf8;">🔭 SPHEREx SkyBlink <span style="font-size:13px; color:#7dd3fc; background:rgba(56,189,248,0.14); padding:2px 8px; border-radius:6px;">v6.0 SCIENCE CONSOLE</span></div>
      <div style="font-size:13px; color:#cbd5e1; margin-top:4px;">{t['subtitle']}</div>
    </div>
    """, unsafe_allow_html=True)
with top_col2:
    st.markdown("<div style='text-align:center; padding-top:22px; color:#7dd3fc;'><code>NASA IRSA • ESA GAIA DR3 • CSA CADC</code></div>", unsafe_allow_html=True)

default_target = qp.get("target", "Barnard's Star")
target_keys = list(NASA_BENCHMARK_TARGETS.keys()) + ["__CUSTOM__"]
if default_target not in NASA_BENCHMARK_TARGETS and default_target != "__CUSTOM__":
    default_target = "Barnard's Star"
default_fov = float(qp.get("fov", 3.0)) if str(qp.get("fov", "3.0")).replace(".", "", 1).isdigit() else 3.0
default_release = qp.get("release", "all")
if default_release not in SPHEREX_COLLECTIONS:
    default_release = "all"

bar1, bar2, bar3, bar4 = st.columns([2.0, 2.3, 1.1, 1.1])
with bar1:
    target_key = st.selectbox(
        t["target_label"],
        target_keys,
        index=target_keys.index(default_target),
        format_func=lambda k: t["custom_option"] if k == "__CUSTOM__" else k,
    )

with bar2:
    if target_key == "__CUSTOM__":
        c_ra, c_dec = st.columns(2)
        ra_in = c_ra.text_input(t["ra_label"], value=qp.get("ra", "08h55m10.831s"))
        dec_in = c_dec.text_input(t["dec_label"], value=qp.get("dec", "-07d14m42.53s"))
        target_desc = t["custom_desc"]
    else:
        tinfo = NASA_BENCHMARK_TARGETS[target_key]
        ra_in, dec_in = tinfo["ra"], tinfo["dec"]
        target_desc = t["target_descs"][target_key]
        st.text_input(t["active_coords"], value=f"{ra_in} , {dec_in}", disabled=True)
        st.caption(target_desc)

with bar3:
    fov_options = [1.5, 3.0, 6.0, 12.0]
    closest_fov = min(fov_options, key=lambda x: abs(x - default_fov))
    fov_choice = st.select_slider(t["fov_label"], options=fov_options, value=closest_fov)

with bar4:
    release_choice = st.selectbox(
        t["release_label"],
        SPHEREX_COLLECTIONS,
        index=SPHEREX_COLLECTIONS.index(default_release),
        format_func=lambda c: "All (DR1 + QR3)" if c == "all" else c,
    )

ctrl_col, view_col = st.columns([1.25, 2.75])

with ctrl_col:
    st.markdown(f"#### `{t['sec_data']}`")
    source_key = st.selectbox(
        t["source_label"],
        ["spherex_plus_wise", "spherex_only", "wise_w2", "wise_w1"],
        format_func=lambda k: t["sources"][k],
    )
    subtract_zodi = st.checkbox(t["zodi_label"], value=True)
    mask_flagged = st.checkbox(t["mask_flags_label"], value=False)
    max_spherex = st.slider(t["max_spherex_label"], 4, 16, 8)

    st.markdown(f"#### `{t['sec_mode']}`")
    mode_key = st.radio(
        t["sec_mode"],
        ["blink", "diff", "coadd", "arrays", "side", "aladin"],
        format_func=lambda k: t["modes"][k],
        label_visibility="collapsed",
    )

    st.markdown(f"#### `{t['sec_stretch']}`")
    stretch_key = st.selectbox(t["stretch_label"], ["asinh", "log", "sqrt", "linear"], format_func=lambda k: t["stretches"][k])
    invert_cmap = st.checkbox(t["invert_label"], value=False)
    psf_match = st.checkbox(t["psf_match_label"], value=False)
    show_gaia_overlay = st.checkbox(t["gaia_label"], value=True)
    pm_horizon = st.slider(t["pm_label"], 1, 25, 10)
    show_sed = st.checkbox(t["sed_label"], value=True)
    show_cadc = st.checkbox(t["cadc_label"], value=True)

    st.markdown("---")
    load_btn = st.button(t["query_btn"], type="primary", use_container_width=True)
    if st.button(t["refresh_btn"], use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("sky_payload", None)
        st.session_state.pop("last_query_sig", None)
        st.rerun()

with view_col:
    try:
        sky_coord = parse_sky_coord(ra_in, dec_in)
    except Exception as e:
        st.error(f"{t['coord_err']} {e}")
        st.stop()

    wise_band = "2" if (source_key == "wise_w2" or target_key == "WISE 0855") else "1"
    current_sig = (
        target_key,
        round(sky_coord.ra.deg, 5),
        round(sky_coord.dec.deg, 5),
        fov_choice,
        release_choice,
        source_key,
        wise_band,
        subtract_zodi,
        mask_flagged,
        max_spherex,
    )
    need_fetch = (
        "sky_payload" not in st.session_state
        or (target_key != "__CUSTOM__" and st.session_state.get("last_query_sig") != current_sig)
    )

    if load_btn or need_fetch:
        st.query_params["lang"] = lang
        st.query_params["target"] = target_key
        st.query_params["ra"] = f"{sky_coord.ra.deg:.5f}"
        st.query_params["dec"] = f"{sky_coord.dec.deg:.5f}"
        st.query_params["fov"] = str(fov_choice)
        st.query_params["release"] = release_choice

        with st.status(t["loading_title"], expanded=True) as status:
            want_spherex = source_key in ("spherex_only", "spherex_plus_wise")
            want_wise = source_key in ("spherex_plus_wise", "wise_w1", "wise_w2")

            bundle = cached_archive_bundle(
                sky_coord.ra.deg,
                sky_coord.dec.deg,
                fov_choice,
                release_choice,
                max_spherex,
                subtract_zodi,
                mask_flagged,
                want_spherex,
                want_wise,
                wise_band,
                show_sed,
                show_cadc,
            )
            spherex_frames = bundle["spherex_frames"]
            wise_frames = bundle["wise_frames"]
            gaia_df = bundle["gaia_df"]
            sed_df = bundle["sed_df"]
            cadc_df = bundle["cadc_df"]
            diagnostics = list(bundle["diagnostics"])

            raw_frames = spherex_frames + wise_frames
            raw_frames.sort(key=lambda f: f.get("mjd", 0.0))

            ref_wcs = build_reference_wcs(sky_coord, fov_choice, npix=160)
            aligned_frames = []
            for fr in raw_frames:
                al = align_frame_safely(fr, ref_wcs, shape_out=(160, 160))
                if al is not None:
                    aligned_frames.append(al)

            spx_aligned = [f for f in aligned_frames if "SPHEREx" in f.get("band", "") or "SPHEREx" in f.get("mission", "")]
            spx_spec_df = extract_calibrated_spherex_spectrum(spx_aligned, sky_coord) if spx_aligned else pd.DataFrame()

            st.session_state["sky_payload"] = {
                "coord": sky_coord,
                "fov": fov_choice,
                "release": release_choice,
                "ref_wcs": ref_wcs,
                "frames": aligned_frames,
                "spherex_frames": spx_aligned,
                "wise_frames": [f for f in aligned_frames if f not in spx_aligned],
                "spx_spec_df": spx_spec_df,
                "gaia_df": gaia_df,
                "sed_df": sed_df,
                "cadc_df": cadc_df,
                "diagnostics": diagnostics,
            }
            st.session_state["last_query_sig"] = current_sig
            status.update(
                label=t["loading_done"].format(
                    n_spherex=len(spx_aligned),
                    n_wise=len(aligned_frames) - len(spx_aligned),
                    n_gaia=len(gaia_df),
                    n_neossat=len(cadc_df),
                ),
                state="complete",
                expanded=False,
            )
    else:
        st.query_params["lang"] = lang

    payload = st.session_state.get("sky_payload")
    if payload:
        frames = payload["frames"]
        spx_frames = payload["spherex_frames"]
        gaia_df = payload["gaia_df"]
        cadc_df = payload["cadc_df"]
        cmap_choice = "gray_r" if invert_cmap else "magma"

        max_pm = 0.0
        if gaia_df is not None and len(gaia_df) > 0 and "pmra" in gaia_df.columns and "pmdec" in gaia_df.columns:
            pm_tot = np.hypot(
                pd.to_numeric(gaia_df["pmra"], errors="coerce").fillna(0.0).values,
                pd.to_numeric(gaia_df["pmdec"], errors="coerce").fillna(0.0).values,
            )
            max_pm = float(np.max(pm_tot)) if pm_tot.size > 0 else 0.0

        sampled_waves = sorted({round(f["wavelength_um"], 3) for f in spx_frames if f.get("wavelength_um") is not None})
        sampled_arrays = sorted({f["band"] for f in spx_frames})

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>{t['kpi_coords']}</div><div class='kpi-val'>{payload['coord'].ra.deg:.4f}°, {payload['coord'].dec.deg:+.4f}°</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>{t['kpi_spherex']}</div><div class='kpi-val'>{len(spx_frames)} ({len(sampled_arrays)}/6 Arrays)</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>{t['kpi_channels']}</div><div class='kpi-val'>{len(sampled_waves)} λ Channels</div></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi-card'><div class='kpi-lbl'>{t['kpi_pm']}</div><div class='kpi-val'>{max_pm:.1f} mas/yr</div></div>", unsafe_allow_html=True)

        share_qs = urllib.parse.urlencode({
            "lang": lang,
            "target": target_key,
            "ra": f"{payload['coord'].ra.deg:.5f}",
            "dec": f"{payload['coord'].dec.deg:.5f}",
            "fov": str(payload["fov"]),
            "release": payload["release"],
        })
        st.caption(f"🔗 {t['share_label']} `?{share_qs}`")

        if frames:
            with st.expander(f"🕒 {t['timeline_title']} ({len(frames)})", expanded=False):
                tl_df = pd.DataFrame([
                    {
                        "Order": i + 1,
                        "DATE-OBS": fr.get("date_obs") or f"~{fr.get('jyear', 0):.1f}",
                        "MJD": round(float(fr.get("mjd", 0.0)), 4),
                        "Mission / Collection": fr.get("mission", ""),
                        "Band / Array": fr.get("band", ""),
                        "λ (µm)": round(float(fr["wavelength_um"]), 4) if fr.get("wavelength_um") else None,
                        "Unit": fr.get("bunit", ""),
                    }
                    for i, fr in enumerate(frames)
                ])
                st.dataframe(tl_df, use_container_width=True, hide_index=True)

        if mode_key == "aladin":
            aladin_html = f"""
            <div id="aladin-lite-div" style="width:100%; height:520px; border-radius:10px; border:1px solid rgba(56,189,248,0.35);"></div>
            <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
            <script>
              A.init.then(() => {{
                A.aladin('#aladin-lite-div', {{
                  survey: 'P/allWISE/color',
                  target: '{payload["coord"].ra.deg:.6f} {payload["coord"].dec.deg:+.6f}',
                  fov: {payload["fov"] / 60.0:.4f},
                  showReticle: true
                }});
              }});
            </script>
            """
            components.html(aladin_html, height=540)

        elif mode_key == "arrays":
            st.markdown(f"#### `{t['arrays_title']}`")
            st.info(t["arrays_explain"])
            summary_df = build_channel_summary_table(spx_frames)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            cols_grid = st.columns(3)
            for idx_b, arr in enumerate(SPHEREX_DETECTOR_ARRAYS):
                with cols_grid[idx_b % 3]:
                    matched = [fr for fr in spx_frames if fr.get("band") == arr["id"]]
                    if matched:
                        same_s = [m for m in matched if m["aligned"].shape == matched[0]["aligned"].shape]
                        coadd_d = np.nanmedian(np.stack([m["aligned"] for m in same_s], axis=0), axis=0)
                        b64 = render_frame_to_base64(
                            apply_stretch(coadd_d, stretch_key),
                            f"{arr['title']} ({len(same_s)} obs)",
                            cmap_choice,
                            gaia_df,
                            same_s[0]["used_wcs"],
                            show_gaia_overlay,
                            frame_jyear=float(np.mean([m["jyear"] for m in same_s])),
                            pm_years=pm_horizon,
                        )
                        st.image(f"data:image/png;base64,{b64}", use_container_width=True)
                    else:
                        st.warning(f"**{arr['title']}**\n\n{t['arrays_none']}")

            if spx_frames:
                st.markdown(f"#### `{t['channel_explorer']}`")
                ch_sorted = sorted(spx_frames, key=lambda x: x.get("wavelength_um") or 99.0)
                ch_idx = st.selectbox(
                    "SPHEREx LVF Channel Frame:",
                    range(len(ch_sorted)),
                    format_func=lambda i: f"{ch_sorted[i]['label']} | {ch_sorted[i].get('wave_source', '')}",
                )
                ch_fr = ch_sorted[ch_idx]
                b64_ch = render_frame_to_base64(
                    apply_stretch(ch_fr["aligned"], stretch_key),
                    f"{ch_fr['label']} [{ch_fr.get('bunit', '')}]",
                    cmap_choice,
                    gaia_df,
                    ch_fr["used_wcs"],
                    show_gaia_overlay,
                    frame_jyear=ch_fr.get("jyear", 2025.5),
                    pm_years=pm_horizon,
                )
                st.image(f"data:image/png;base64,{b64_ch}", use_container_width=False)

        elif not frames:
            st.warning(t["no_frames"])

        elif mode_key == "blink":
            encoded = []
            for fr in frames:
                b64 = render_frame_to_base64(
                    apply_stretch(fr["aligned"], stretch_key),
                    fr["label"],
                    cmap_choice,
                    gaia_df,
                    fr["used_wcs"],
                    show_gaia_overlay,
                    frame_jyear=fr.get("jyear", 2020.0),
                    pm_years=pm_horizon,
                )
                meta_line = f"{fr.get('mission', '')} | Unit: {fr.get('bunit', '')} | {fr.get('flags', '')}"
                encoded.append({"label": fr["label"], "meta": meta_line, "b64": b64})
            render_html_blink_player(encoded, t["player"], default_fps=2.0)

        elif mode_key == "diff":
            same_grid = [fr for fr in frames if fr["aligned"].shape == frames[0]["aligned"].shape]
            if len(same_grid) >= 2:
                ep_labels = [f"{i+1}. {fr['label']} [{fr.get('bunit', '')}]" for i, fr in enumerate(same_grid)]
                dc1, dc2 = st.columns(2)
                idx_a = dc1.selectbox(t["diff_a"], range(len(same_grid)), index=0, format_func=lambda i: ep_labels[i])
                idx_b = dc2.selectbox(t["diff_b"], range(len(same_grid)), index=len(same_grid) - 1, format_func=lambda i: ep_labels[i])

                diff_res = compute_difference_and_candidates(
                    same_grid[idx_a],
                    same_grid[idx_b],
                    psf_smooth_sigma=1.0 if psf_match else 0.0,
                    snr_threshold=4.0,
                )
                if diff_res["warnings"]:
                    st.warning(f"{t['diff_unit_warn']} ({'; '.join(diff_res['warnings'])})")

                diff_arr = diff_res["diff"]
                abs_vals = np.abs(diff_arr[np.isfinite(diff_arr)])
                vlim = float(np.percentile(abs_vals, 98.5)) if abs_vals.size > 0 else 1.0
                if vlim <= 0:
                    vlim = 1.0

                fig = plt.figure(figsize=(5.6, 5.1), facecolor="#050814")
                ax = fig.add_subplot(111, projection=diff_res["wcs"])
                ax.set_facecolor("#050814")
                ax.imshow(diff_arr, origin="lower", cmap="coolwarm", vmin=-vlim, vmax=vlim)
                if diff_res["candidates"]:
                    cx = [c["x_px"] for c in diff_res["candidates"]]
                    cy = [c["y_px"] for c in diff_res["candidates"]]
                    ax.scatter(cx, cy, s=130, edgecolor="#22c55e", facecolor="none", linewidths=1.5, linestyle="--", label=">4σ Candidate")
                    ax.legend(loc="upper right", fontsize=7.5, facecolor="#0b1022", labelcolor="#22c55e")
                ax.set_title(f"Diff: ({same_grid[idx_b]['label']}) − ({same_grid[idx_a]['label']})", color="#38bdf8", fontsize=8.8)
                ax.coords[0].set_axislabel("RA (ICRS)", color="#94a3b8", fontsize=8)
                ax.coords[1].set_axislabel("Dec (ICRS)", color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#94a3b8", labelsize=7)
                ax.grid(color="#1e293b", ls=":", alpha=0.65)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(t["diff_caption"])

                st.markdown(f"##### `{t['moving_title']}`")
                if diff_res["candidates"]:
                    st.dataframe(pd.DataFrame(diff_res["candidates"]), use_container_width=True, hide_index=True)
                else:
                    st.caption(t["moving_none"])
            else:
                st.info(t["diff_need_two"])

        elif mode_key == "coadd":
            same_grid = [fr for fr in frames if fr["aligned"].shape == frames[0]["aligned"].shape]
            group_labels = sorted({f"{fr.get('band', 'Unknown')} • {fr.get('bunit', '')}" for fr in same_grid})
            selected_grp = group_labels[0]
            if len(group_labels) > 1:
                selected_grp = st.selectbox("Select Compatible Band & Unit Group to Stack:", group_labels)
            stack_pool = [fr for fr in same_grid if f"{fr.get('band', 'Unknown')} • {fr.get('bunit', '')}" == selected_grp]
            method = st.radio(t["coadd_method"], ["Median", "Mean"], horizontal=True)
            cube = np.stack([fr["aligned"] for fr in stack_pool], axis=0)
            coadd = np.nanmedian(cube, axis=0) if method == "Median" else np.nanmean(cube, axis=0)
            mean_yr = float(np.mean([fr.get("jyear", 2020.0) for fr in stack_pool]))
            b64 = render_frame_to_base64(
                apply_stretch(coadd, stretch_key),
                f"{method} Coadd ({len(stack_pool)} frames • {selected_grp})",
                cmap_choice,
                gaia_df,
                stack_pool[0]["used_wcs"],
                show_gaia_overlay,
                frame_jyear=mean_yr,
                pm_years=pm_horizon,
            )
            st.image(f"data:image/png;base64,{b64}", use_container_width=True)
            st.caption(t["coadd_caption"])

        elif mode_key == "side":
            c1, c2 = st.columns(2)
            for col_ui, fr in zip([c1, c2], [frames[0], frames[-1]]):
                with col_ui:
                    b64 = render_frame_to_base64(
                        apply_stretch(fr["aligned"], stretch_key),
                        fr["label"],
                        cmap_choice,
                        gaia_df,
                        fr["used_wcs"],
                        show_gaia_overlay,
                        frame_jyear=fr.get("jyear", 2020.0),
                        pm_years=pm_horizon,
                    )
                    st.image(f"data:image/png;base64,{b64}", use_container_width=True)

        if frames:
            with st.expander(t["meta_expander"], expanded=False):
                meta_df = pd.DataFrame([
                    {
                        "Obs ID": fr.get("obs_id", ""),
                        "Mission": fr.get("mission", ""),
                        "Array / Band": fr.get("band", ""),
                        "λ (µm)": round(float(fr["wavelength_um"]), 4) if fr.get("wavelength_um") else None,
                        "λ Source": fr.get("wave_source", ""),
                        "MJD": round(float(fr.get("mjd", 0.0)), 4),
                        "Unit": fr.get("bunit", ""),
                        "Zodi (MJy/sr)": round(float(fr["zodi_median"]), 4) if fr.get("zodi_median") is not None else None,
                        "Flags": fr.get("flags", ""),
                    }
                    for fr in frames
                ])
                st.dataframe(meta_df, use_container_width=True, hide_index=True)

        if payload["diagnostics"]:
            with st.expander(t["diag_expander"], expanded=False):
                for line in payload["diagnostics"]:
                    st.code(line)

        if show_sed:
            st.markdown(f"#### `{t['sed_title']}`")
            spx_spec_df = payload["spx_spec_df"]
            sed_df = payload["sed_df"]

            if len(spx_spec_df) > 0 or len(sed_df) > 0:
                fig_s, ax_s = plt.subplots(figsize=(9.0, 2.9), facecolor="#050814")
                ax_s.set_facecolor("#0b1022")

                if len(sed_df) > 0:
                    sed_sorted = sed_df.sort_values("wavelength_um")
                    ax_s.errorbar(
                        sed_sorted["wavelength_um"],
                        sed_sorted["flux_mjy"],
                        yerr=sed_sorted["sigma_mjy"] if "sigma_mjy" in sed_sorted.columns else None,
                        fmt="o--",
                        color="#38bdf8",
                        linewidth=1.4,
                        markersize=5,
                        capsize=3,
                        label="IRSA 2MASS/AllWISE (mJy)",
                    )
                    for _, r in sed_sorted.iterrows():
                        ax_s.annotate(r["band"], (r["wavelength_um"], r["flux_mjy"]), textcoords="offset points", xytext=(0, 7), ha="center", color="#cbd5e1", fontsize=7.5)

                if len(spx_spec_df) > 0:
                    ax_s.errorbar(
                        spx_spec_df["wavelength_um"],
                        spx_spec_df["flux_mjy"],
                        yerr=spx_spec_df["sigma_mjy"],
                        fmt="s-",
                        color="#facc15",
                        linewidth=1.8,
                        markersize=6,
                        capsize=3,
                        label="SPHEREx L2 Calibrated Aperture (MJy/sr→mJy ± σ)",
                    )

                ymax = max(ax_s.get_ylim()[1], 1e-6)
                for mol_x, mol_lbl in MOLECULAR_MARKERS:
                    ax_s.axvline(mol_x, color="#94a3b8", linestyle=":", alpha=0.45)
                    ax_s.text(mol_x + 0.03, ymax * 0.85, mol_lbl, color="#7dd3fc", fontsize=7.5)

                ax_s.set_xlim(0.7, 5.3)
                ax_s.set_xlabel(t["sed_xlabel"], color="#94a3b8", fontsize=8)
                ax_s.set_ylabel(t["sed_ylabel"], color="#94a3b8", fontsize=8)
                ax_s.tick_params(colors="#94a3b8", labelsize=8)
                ax_s.legend(loc="upper right", fontsize=7.5, facecolor="#0b1022", labelcolor="#e2e8f0")
                ax_s.grid(color="#1e293b", linestyle=":", linewidth=0.5)
                st.pyplot(fig_s)
                plt.close(fig_s)

                if len(spx_spec_df) > 0:
                    st.dataframe(
                        spx_spec_df[["band", "wavelength_um", "bandwidth_um", "flux_mjy", "sigma_mjy", "wave_source", "source"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                if payload["wise_frames"]:
                    st.caption(t["sed_note_unwise"])
            else:
                st.info(t["sed_empty"])

        if gaia_df is not None and len(gaia_df) > 0:
            st.markdown(f"#### `{t['gaia_title']}`")
            st.dataframe(gaia_df, use_container_width=True, hide_index=True)

        if show_cadc:
            st.markdown(f"#### `{t['cadc_title']}`")
            if len(cadc_df) > 0:
                st.dataframe(cadc_df, use_container_width=True, hide_index=True)
            else:
                st.caption(t["cadc_empty"])