# -*- coding: utf-8 -*-
"""
data_collection.py
--------------------------------
اسکریپت جمع‌آوری داده واقعی برای فاز آموزش مدل.
"""

import json
import time

try:
    import requests
except ImportError:
    requests = None


WIKI_API = "https://fa.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "PersianTextSummarizationProject/1.0"}

with open("wiki_titles.json", encoding="utf-8") as f:
    WIKI_SAMPLE_TITLES = json.load(f)

NEWS_RSS_FEEDS = {
    "irna": "https://www.irna.ir/rss",
    "isna": "https://www.isna.ir/rss",
}


def fetch_wikipedia_article(title: str, max_retries: int = 3) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
    }

    for attempt in range(max_retries):
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            wait_time = 10 * (attempt + 1)
            print(f"  ({title}: محدودیت نرخ، {wait_time} ثانیه صبر می‌کنم...)")
            time.sleep(wait_time)
            continue
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        full_text = page.get("extract", "")
        summary = full_text.split("\n\n")[0] if full_text else ""
        return {
            "id": f"wiki_{title}",
            "source": "fa.wikipedia.org",
            "title": title,
            "text": full_text,
            "summary": summary,
        }

    raise RuntimeError("بعد از چند بار تلاش هم موفق نشد")


def collect_wikipedia_corpus(titles, out_path="raw_corpus_wiki.jsonl"):
    if requests is None:
        raise RuntimeError("کتابخانه requests نصب نیست: pip install requests")

    with open(out_path, "w", encoding="utf-8") as f:
        for title in titles:
            try:
                record = fetch_wikipedia_article(title)
                if len(record["text"].split()) >= 500:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"OK: {title} ({len(record['text'].split())} کلمه)")
                else:
                    print(f"SKIP (کوتاه): {title}")
            except Exception as e:
                print(f"ERROR در {title}: {e}")
            time.sleep(3)


if __name__ == "__main__":
    print("شروع جمع‌آوری داده از ویکی‌پدیای فارسی...")
    collect_wikipedia_corpus(WIKI_SAMPLE_TITLES)
    print("\nپایان جمع‌آوری.")