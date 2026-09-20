# -*- coding: utf-8 -*-
"""
persian_preprocessing.py
--------------------------------
پایپ‌لاین پیش‌پردازش متن برای پروژه «خلاصه‌سازی خودکار متون طولانی فارسی و انگلیسی»
گام ۳ پروپوزال: آماده‌سازی و پردازش داده‌های متنی

این ماژول به‌صورت مستقل (بدون وابستگی به کتابخانه‌های خارجی مانند Hazm) نوشته شده
تا در هر محیطی (حتی بدون اتصال اینترنت برای نصب پکیج) قابل اجرا باشد.
اگر Hazm در محیط اجرای نهایی نصب باشد، تابع‌های sentence_segment و tokenize
به‌سادگی می‌توانند با معادل Hazm جایگزین شوند (نکته در پایان فایل).
"""

import re
import json
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict


# ----------------------------------------------------------------------
# ۱. نرمال‌سازی متن
# ----------------------------------------------------------------------

# یکسان‌سازی کاراکترهای عربی/فارسی که در متون وب به‌کرات با هم قاطی می‌شوند
_CHAR_MAP = {
    "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "‌": " ",   # نیم‌فاصله -> فاصله معمولی (برای ساده‌سازی توکن‌سازی)
    "ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "آ",
    "ؤ": "و", "ئ": "ی",
}

# اعداد فارسی/عربی -> اعداد انگلیسی (برای یکنواختی در پردازش)
_DIGIT_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

# اعراب و علائم دیاکریتیک عربی/فارسی که باید حذف شوند
_DIACRITICS = re.compile(
    "[\u064B-\u0652\u0670\u0640]"  # فتحه، ضمه، کسره، تنوین، سکون، تشدید، کشیده
)

_MULTI_SPACE = re.compile(r"\s+")
_MULTI_PUNCT = re.compile(r"([!؟?.،,]){2,}")


def normalize_text(text: str) -> str:
    """نرمال‌سازی کاراکترها، اعداد، فاصله‌ها و علائم نگارشی تکراری."""
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)  # حذف تیترهای بخش ویکی‌پدیا مثل == تاریخ ==
    text = unicodedata.normalize("NFC", text)

    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    for src, dst in _DIGIT_MAP.items():
        text = text.replace(src, dst)

    text = _MULTI_PUNCT.sub(r"\1", text)     # !!! -> !
    text = _MULTI_SPACE.sub(" ", text)       # فاصله‌های تکراری
    return text.strip()


# ----------------------------------------------------------------------
# ۲. جمله‌بندی (Sentence Segmentation)
# ----------------------------------------------------------------------

# مرزهای پایان جمله در فارسی: . ! ؟ و گاهی «:» قبل از فهرست
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!؟?])\s+")

# اختصارات رایج که نباید باعث شکست جمله شوند (مثلاً "ه.ش." یا "دکتر")
_ABBREVIATIONS = {"ه.ش", "ه.ق", "م.", "دکتر", "آقای", "خانم", "ص.", "ج."}


def sentence_segment(text: str) -> List[str]:
    """تقسیم متن نرمال‌شده به جملات مستقل."""
    raw_sentences = _SENTENCE_BOUNDARY.split(text)
    sentences: List[str] = []
    buffer = ""
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # اگر جمله‌ی قبلی به یک اختصار ختم شده، آن را با جمله بعدی ادغام کن
        if buffer and any(buffer.rstrip().endswith(ab) for ab in _ABBREVIATIONS):
            buffer = f"{buffer} {s}"
            continue
        if buffer:
            sentences.append(buffer)
        buffer = s
    if buffer:
        sentences.append(buffer)
    return sentences


# ----------------------------------------------------------------------
# ۳. توکن‌سازی (Tokenization)
# ----------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[آ-یA-Za-z0-9]+")


def tokenize(sentence: str) -> List[str]:
    """تبدیل یک جمله به لیست توکن‌های کلمه‌ای (حروف فارسی/انگلیسی و اعداد)."""
    return _TOKEN_PATTERN.findall(sentence)


# ----------------------------------------------------------------------
# ۴. حذف کلمات ایست (Stop Words Removal)
# ----------------------------------------------------------------------

PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای",
    "آن", "یک", "شد", "خود", "تا", "کرد", "بر", "هم", "نیز", "یا",
    "اما", "اگر", "چون", "بود", "شده", "می", "های", "هایی", "کند",
    "شود", "دارد", "داشت", "کنند", "دهد", "پس", "چه", "همه", "هر",
    "بین", "بی", "دیگر", "باید", "ما", "شما", "او", "آنها", "من",
    "چند", "چگونه", "کجا", "چرا", "بی‌", "روی"
, "زیر", "بالای",    "یکی", "بوده", "بودند", "شده", "کرده", "دارند", "کنیم", "شوند",
}


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in PERSIAN_STOPWORDS]


# ----------------------------------------------------------------------
# ۵. پایپ‌لاین کامل
# ----------------------------------------------------------------------

@dataclass
class ProcessedDocument:
    doc_id: str
    original_text: str
    normalized_text: str
    sentences: List[str]
    tokens_per_sentence: List[List[str]] = field(default_factory=list)
    tokens_clean_per_sentence: List[List[str]] = field(default_factory=list)
    num_sentences: int = 0
    num_tokens: int = 0
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "sentences": self.sentences,
            "tokens_per_sentence": self.tokens_per_sentence,
            "tokens_clean_per_sentence": self.tokens_clean_per_sentence,
            "num_sentences": self.num_sentences,
            "num_tokens": self.num_tokens,
            "summary": self.summary,
        }


def preprocess_document(doc_id: str, text: str, summary: str = "") -> ProcessedDocument:
    normalized = normalize_text(text)
    sentences = sentence_segment(normalized)
    tokens_per_sentence = [tokenize(s) for s in sentences]
    tokens_clean_per_sentence = [remove_stopwords(t) for t in tokens_per_sentence]
    num_tokens = sum(len(t) for t in tokens_per_sentence)

    return ProcessedDocument(
        doc_id=doc_id,
        original_text=text,
        normalized_text=normalized,
        sentences=sentences,
        tokens_per_sentence=tokens_per_sentence,
        tokens_clean_per_sentence=tokens_clean_per_sentence,
        num_sentences=len(sentences),
        num_tokens=num_tokens,
        summary=summary,
    )


def process_corpus(records: List[Dict], out_path: str) -> List[ProcessedDocument]:
    """
    records: [{"id": ..., "text": ..., "summary": ...}, ...]
    خروجی را به‌صورت JSON در out_path ذخیره می‌کند.
    """
    processed = [preprocess_document(r["id"], r["text"], r.get("summary", "")) for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in processed], f, ensure_ascii=False, indent=2)
    return processed


# ----------------------------------------------------------------------
# نکته درباره‌ی Hazm
# ----------------------------------------------------------------------
# در محیط توسعه نهایی (که به اینترنت دسترسی دارد) پیشنهاد می‌شود:
#   pip install hazm
#   from hazm import Normalizer, SentenceTokenizer, WordTokenizer, stopwords_list
# و توابع normalize_text / sentence_segment / tokenize / remove_stopwords
# با معادل‌های Hazm جایگزین شوند تا دقت ریشه‌یابی و برچسب‌گذاری دستوری
# (POS Tagging) نیز به پایپ‌لاین اضافه گردد. رابط (ورودی/خروجی) این ماژول
# طوری طراحی شده که این جایگزینی بدون تغییر در بقیه‌ی سیستم ممکن باشد.
