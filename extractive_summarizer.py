# -*- coding: utf-8 -*-
"""
extractive_summarizer.py
--------------------------------
لایه اول معماری پروژه: خلاصه‌سازی استخراجی با TF-IDF
طبق پروپوزال: امتیازدهی بر اساس سه معیار
  1) فراوانی واژگان (TF-IDF)
  2) موقعیت جمله در متن
  3) ارتباط معنایی با سایر جملات (شباهت کسینوسی)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from persian_preprocessing import tokenize, remove_stopwords, ENGLISH_STOPWORDS
def score_sentences(sentences, position_weight=0.15, similarity_weight=0.25):
    """
    به هر جمله یک امتیاز نهایی می‌دهد (ترکیب سه معیار).
    sentences: لیستی از رشته‌های متنی (هر رشته یک جمله)
    """
    if len(sentences) < 2:
        return np.ones(len(sentences))

    # ---------- معیار ۱: TF-IDF ----------
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # امتیاز هر جمله = میانگین مقادیر TF-IDF کلمات آن جمله
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
    # نرمال‌سازی به بازه ۰ تا ۱
    if tfidf_scores.max() > 0:
        tfidf_scores = tfidf_scores / tfidf_scores.max()

    # ---------- معیار ۲: موقعیت جمله ----------
    # جملات ابتدای متن امتیاز بیشتری می‌گیرند (کاهش خطی)
    n = len(sentences)
    position_scores = np.array([1.0 - (i / n) for i in range(n)])

    # ---------- معیار ۳: شباهت با سایر جملات ----------
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # امتیاز هر جمله = میانگین شباهتش با همه جملات دیگر
    similarity_scores = similarity_matrix.mean(axis=1)
    if similarity_scores.max() > 0:
        similarity_scores = similarity_scores / similarity_scores.max()

    # ---------- ترکیب نهایی سه معیار ----------
    tfidf_weight = 1.0 - position_weight - similarity_weight
    final_scores = (
        tfidf_weight * tfidf_scores
        + position_weight * position_scores
        + similarity_weight * similarity_scores
    )
    return final_scores
def extractive_summarize(sentences, ratio=0.2):
    """
    sentences: لیست جملات (خروجی sentence_segment از ماژول قبلی)
    ratio: نسبت خلاصه به متن اصلی (طبق پروپوزال بین ۰.۱ تا ۰.۳)
    """
    if len(sentences) <= 2:
        return sentences

    scores = score_sentences(sentences)

    n_selected = max(1, int(len(sentences) * ratio))

    # اندیس جملاتی که بالاترین امتیاز را دارند
    top_indices = np.argsort(scores)[-n_selected:]
    # مرتب‌سازی بر اساس ترتیب اصلی متن (نه ترتیب امتیاز) تا خلاصه منسجم بماند
    top_indices_sorted = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return summary_sentences
def extract_keywords(sentences, top_n=8):
    """
    استخراج مهم‌ترین کلمات کلیدی متن، با حذف کلمات ایست فارسی و انگلیسی.
    """
    if len(sentences) < 2:
        return []

    cleaned_sentences = []
    for s in sentences:
        tokens = tokenize(s)
        clean_tokens = remove_stopwords(tokens)
        clean_tokens = [t for t in clean_tokens if t.lower() not in ENGLISH_STOPWORDS]
        cleaned_sentences.append(" ".join(clean_tokens))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()

    term_scores = list(zip(terms, scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)

    keywords = [term for term, score in term_scores[:top_n] if len(term) > 1]
    return keywords
if __name__ == "__main__":
    # تست سریع با یک متن نمونه
    test_sentences = [
        "هوش مصنوعی یکی از شاخه‌های علوم کامپیوتر است.",
        "این حوزه به مطالعه سیستم‌هایی می‌پردازد که می‌توانند هوشمندانه رفتار کنند.",
        "امروزه هوش مصنوعی در بسیاری از صنایع کاربرد دارد.",
        "یادگیری ماشین یکی از زیرشاخه‌های مهم هوش مصنوعی است.",
        "در یادگیری ماشین، مدل‌ها از داده‌ها الگو یاد می‌گیرند.",
        "هوا امروز آفتابی بود و باران نبارید.",
        "کاربردهای هوش مصنوعی شامل تشخیص تصویر و پردازش زبان طبیعی است.",
    ]

    summary = extractive_summarize(test_sentences, ratio=0.4)
    print("خلاصه تولیدشده:")
    for s in summary:
        print(f"  - {s}")