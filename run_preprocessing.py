import json
from persian_preprocessing import process_corpus

records = []
with open("raw_corpus_wiki.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            records.append({"id": r["id"], "text": r["text"], "summary": r.get("summary", "")})

print(f"تعداد رکوردهای خام: {len(records)}")

processed = process_corpus(records, "final_processed_corpus.json")

print(f"پردازش {len(processed)} سند با موفقیت تمام شد.")
print(f"خروجی در final_processed_corpus.json ذخیره شد.")

total_sentences = sum(p.num_sentences for p in processed)
total_tokens = sum(p.num_tokens for p in processed)
print(f"مجموع جملات: {total_sentences}")
print(f"مجموع توکن‌ها: {total_tokens}")
print(f"میانگین کلمه هر سند: {total_tokens/len(processed):.1f}")