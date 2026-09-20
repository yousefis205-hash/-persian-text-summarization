import json
from extractive_summarizer import extractive_summarize

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

# اولین سند رو برای تست انتخاب می‌کنیم
doc = data[0]

print(f"عنوان سند: {doc['doc_id']}")
print(f"تعداد جملات اصلی: {doc['num_sentences']}")
print()

summary = extractive_summarize(doc["sentences"], ratio=0.2)

print(f"تعداد جملات خلاصه: {len(summary)}")
print()
print("=== خلاصه تولیدشده توسط الگوریتم ===")
for s in summary:
    print(f"- {s}")

print()
print("=== خلاصه مرجع (چکیده ویکی‌پدیا) ===")
print(doc["summary"])