import json

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

longest_doc = max(data, key=lambda d: d["num_tokens"])

with open("long_test.txt", "w", encoding="utf-8") as f:
    f.write(longest_doc["normalized_text"])

print(f"طولانی‌ترین سند: {longest_doc['doc_id']}")
print(f"تعداد کلمات: {longest_doc['num_tokens']}")
print("در فایل long_test.txt ذخیره شد.")