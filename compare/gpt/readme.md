模型：gpt-4o-mini

experiment_runner.py:
A:一般（prompt+few-shot)
B:加資料（prompt+few-shot+dataset1.jsonl)

experimemt_new.csv:experiment_runner.py的結果

evaluate_comparison2.py(他前面做一個evaluate_comparison.py）：匯入scorer(scorer_sentences.py)

final_scored_report.csv:最後gpt比較結果
