# -*- coding: utf-8 -*-
"""
1vs4sentence.py
---------------------------------
1) 讀取句子庫 (SENTENCES)
2) 呼叫 seprompt.py 取得 Prompt (含 Few-Shot 範例)
3) 產出 outputs/1vs4sentence_chain.csv
   格式：原始句子 | Level1 改寫 | Level2 改寫 | Level3 改寫 | Level4 改寫

【改動重點】
- 原本：每個 level 都用原始句子 s 生成（平行）
- 現在：Level1 -> Level2 -> Level3 -> Level4（階梯式）
"""

import os, csv, time, random
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# 1. 引用 Prompt 模組
# =========================
try:
    from seprompt import build_prompt, LEVELS
except ImportError:
    print("❌ 錯誤：找不到 seprompt.py！")
    exit()

# ============= 參數設定 =============
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
MAX_CHARS = 60                # 每句改寫長度上限
RANDOM_SEED = 42
SLEEP_SEC_BETWEEN_CALLS = 0.1 # 避免顯卡過熱
# ===================================

random.seed(RANDOM_SEED)

# -----------------------------
# 測試句子庫
# -----------------------------
SENTENCES: List[str] = [
    "早安你好",
    "你報告做完了嗎",
    "你衣服穿反了",
    "你化的妝好難看",
    "你剛剛說甚麼",
    "你今天心情不好嗎",
    "你可以幫我個忙嗎",
    "你覺得這樣合理嗎",
    "你怎麼還沒來",
    "這個便當有點冷掉了",
    "你最近是不是胖了",
    "為什麼不回我訊息",
]

# -----------------
# 模型初始化與呼叫
# -----------------
def _init_model():
    print(f"🔄 正在載入模型：{MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("✅ 模型載入完成！")
    return tokenizer, model

def call_model(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    # 套用 Chat Template
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,    # 建議 0.8 最穩定，1.4 會胡言亂語
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # --- 精準截取回覆 (最穩定的寫法) ---
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return resp[:MAX_CHARS]

def generate_rows() -> List[Dict[str, Any]]:
    tokenizer, model = _init_model()
    rows: List[Dict[str, Any]] = []
    total = len(SENTENCES)

    print("🚀 開始批次生成（階梯式 1→2→3→4）...")

    for i, s in enumerate(SENTENCES, start=1):
        print(f"[{i:>2}/{total}] 正在改寫：{s}")

        # ⭐ 階梯式：下一層用上一層輸出當輸入
        current_sentence = s

        for lvl in LEVELS:
            prompt = build_prompt(current_sentence, lvl, max_chars=MAX_CHARS)
            resp = call_model(tokenizer, model, prompt)

            if not resp:
                resp = "(生成失敗)"

            rows.append({
                "sentence": s,      # 保留最原始句子
                "level": lvl,
                "response": resp
            })

            print(f"    -> Lv{lvl}: {resp}")

            # ⭐⭐ 關鍵：把這層輸出當下一層輸入
            current_sentence = resp

            if SLEEP_SEC_BETWEEN_CALLS:
                time.sleep(SLEEP_SEC_BETWEEN_CALLS)

    return rows

def save_csv(rows: List[Dict[str, Any]], out_dir: str = "outputs") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "1vs4sentence_chain.csv")

    # --- 資料轉置 (Pivot) ---
    grouped: Dict[str, Dict[int, str]] = {}
    for r in rows:
        s = r["sentence"]
        grouped.setdefault(s, {})[r["level"]] = r["response"]

    print(f"\n💾 正在寫入 CSV：{csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["原始句子"] + [f"Level {i} 改寫" for i in LEVELS]
        writer.writerow(header)

        for s in SENTENCES:
            lv_map = grouped.get(s, {})
            row = [s] + [lv_map.get(i, "") for i in LEVELS]
            writer.writerow(row)

    print(f"✅ 完成！請打開 {csv_path} 查看結果。")
    return csv_path

if __name__ == "__main__":
    try:
        rows = generate_rows()
        save_csv(rows)
    except Exception as e:
        print("\n❌ 程式執行發生錯誤：")
        print(e)
