import json
from rouge_score import rouge_scorer
from extractive_summarizer import extractive_summarize

with open("final_corpus_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

# روی ۳۰ سند اول تست می‌کنیم (برای سرعت بیشتر)
sample_docs = data[:30]

for doc in sample_docs:
    predicted_sentences = extractive_summarize(doc["sentences"], ratio=0.2)
    predicted_summary = " ".join(predicted_sentences)
    reference_summary = doc["summary"]

    scores = scorer.score(reference_summary, predicted_summary)

    all_scores["rouge1"].append(scores["rouge1"].fmeasure)
    all_scores["rouge2"].append(scores["rouge2"].fmeasure)
    all_scores["rougeL"].append(scores["rougeL"].fmeasure)

print(f"نتایج روی {len(sample_docs)} سند:")
print(f"میانگین ROUGE-1: {sum(all_scores['rouge1'])/len(all_scores['rouge1'])*100:.2f}%")
print(f"میانگین ROUGE-2: {sum(all_scores['rouge2'])/len(all_scores['rouge2'])*100:.2f}%")
print(f"میانگین ROUGE-L: {sum(all_scores['rougeL'])/len(all_scores['rougeL'])*100:.2f}%")
print()
print(f"هدف پروپوزال برای ROUGE-2: بالای 15%")