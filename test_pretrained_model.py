import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

print("در حال دانلود مدل (فقط بار اول طول می‌کشد)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

doc = data[0]
text = doc["normalized_text"][:2000]

print(f"\nعنوان سند: {doc['doc_id']}")
print("\nدر حال تولید خلاصه با مدل...")

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=30,
    num_beams=4,
    no_repeat_ngram_size=2,
)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n=== خلاصه تولیدشده توسط مدل Transformer (بدون آموزش اضافه) ===")
print(summary_text)

print("\n=== خلاصه مرجع (چکیده ویکی‌پدیا) ===")
print(doc["summary"])