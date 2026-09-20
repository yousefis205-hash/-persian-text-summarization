import requests
import json
import time

HEADERS = {"User-Agent": "PersianTextSummarizationProject/1.0"}

def get_category_titles(category, limit=500):
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"رده:{category}",
        "cmlimit": "50",
    }
    url = "https://fa.wikipedia.org/w/api.php"

    while len(titles) < limit:
        resp = None
        for attempt in range(4):
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                break
            wait_time = 10 * (attempt + 1)
            print(f"  (محدودیت نرخ یا خطا، {wait_time} ثانیه صبر می‌کنم...)")
            time.sleep(wait_time)
        else:
            break

        try:
            data = resp.json()
        except Exception:
            break

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            if m["ns"] == 0:
                titles.append(m["title"])
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
        time.sleep(1)

    return titles[:limit]


if __name__ == "__main__":
    categories = [
        "علوم_رایانه", "رایانه", "هوش_مصنوعی", "تاریخ_ایران", "ورزش",
        "جغرافیای_ایران", "پزشکی", "فرهنگ_ایران", "سیاست", "اقتصاد_ایران",
        "هنر", "ادبیات_فارسی", "موسیقی", "سینما", "دانشمندان", "شهرهای_ایران",
        "فیزیک", "شیمی", "زیست‌شناسی", "نجوم",
    ]
    all_titles = []
    for cat in categories:
        t = get_category_titles(cat, limit=100)
        print(f"{cat}: {len(t)} عنوان پیدا شد")
        all_titles.extend(t)
        time.sleep(2)

    all_titles = list(dict.fromkeys(all_titles))

    with open("wiki_titles.json", "w", encoding="utf-8") as f:
        json.dump(all_titles, f, ensure_ascii=False, indent=2)
    print(f"مجموع: {len(all_titles)} عنوان یکتا در wiki_titles.json ذخیره شد")