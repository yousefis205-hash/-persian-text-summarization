from persian_preprocessing import normalize_text, sentence_segment

with open("long_test.txt", encoding="utf-8") as f:
    text = f.read()

normalized = normalize_text(text)
sentences = sentence_segment(normalized)

print(f"تعداد جملات تشخیص داده شده: {len(sentences)}")
print()
for i, s in enumerate(sentences, 1):
    print(f"جمله {i} ({len(s.split())} کلمه): {s[:80]}...")