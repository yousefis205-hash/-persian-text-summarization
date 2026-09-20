import json

with open("final_processed_corpus.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"تعداد کل اسناد قبل از فیلتر: {len(data)}")

result = []
for d in data:
    if d["num_tokens"] < 500:
        continue  # کوتاه‌تر از حد مجاز، قابل استفاده نیست
    if d["num_tokens"] <= 2000:
        result.append(d)  # در بازه مجاز، همینطور نگه دار
    else:
        # بلندتر از حد مجاز: فقط جملات ابتدایی تا سقف ۲۰۰۰ کلمه نگه داشته می‌شود
        kept_sentences = []
        word_count = 0
        for sent, toks in zip(d["sentences"], d["tokens_per_sentence"]):
            if word_count + len(toks) > 2000:
                break
            kept_sentences.append(sent)
            word_count += len(toks)
        if word_count >= 500:
            d["sentences"] = kept_sentences
            d["normalized_text"] = " ".join(kept_sentences)
            d["num_sentences"] = len(kept_sentences)
            d["num_tokens"] = word_count
            result.append(d)

print(f"تعداد اسناد نهایی (بعد از کوتاه‌سازی بلندها): {len(result)}")

with open("final_corpus_filtered.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

lengths = [d["num_tokens"] for d in result]
print(f"میانگین طول نهایی: {sum(lengths)/len(lengths):.1f} کلمه")