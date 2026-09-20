import json
import random

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

# فقط اسنادی که خلاصه‌ی معتبر و غیرخالی دارند
data = [d for d in data if d.get("summary") and len(d["summary"].strip()) > 20]

print(f"تعداد کل اسناد قابل استفاده: {len(data)}")

random.seed(42)
random.shuffle(data)

split_point = int(len(data) * 0.9)
train_data = data[:split_point]
val_data = data[split_point:]

print(f"تعداد اسناد آموزش (train): {len(train_data)}")
print(f"تعداد اسناد اعتبارسنجی (validation): {len(val_data)}")


def to_training_format(records):
    return [
        {"text": r["normalized_text"], "summary": r["summary"]}
        for r in records
    ]


with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(to_training_format(train_data), f, ensure_ascii=False, indent=2)

with open("validation_data.json", "w", encoding="utf-8") as f:
    json.dump(to_training_format(val_data), f, ensure_ascii=False, indent=2)

print("\nفایل‌های train_data.json و validation_data.json ساخته شدند.")