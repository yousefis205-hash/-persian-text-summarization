import json

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

lengths = [d["num_tokens"] for d in data]
sentences = [d["num_sentences"] for d in data]

print(f"تعداد کل اسناد نهایی: {len(data)}")
print(f"مجموع کلمات: {sum(lengths)}")
print(f"مجموع جملات: {sum(sentences)}")
print(f"کمینه طول: {min(lengths)} کلمه")
print(f"بیشینه طول: {max(lengths)} کلمه")
print(f"میانگین طول: {sum(lengths)/len(lengths):.1f} کلمه")

with open("wiki_titles.json", encoding="utf-8") as f:
    titles = json.load(f)
print(f"تعداد کل عناوین جمع‌آوری‌شده (قبل از فیلتر طول اولیه): {len(titles)}")