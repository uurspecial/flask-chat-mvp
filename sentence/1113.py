# -*- coding: utf-8 -*-
"""
change_style.py
---------------------------------
1) 使用句子庫
2) 任務是「根據五級語氣準則改寫句子」（保留原意）。
3) 產出 outputs/change_style.csv，欄位：原始句子、Level1~Level5 改寫。
"""

import os, csv, time, random
from pathlib import Path
from typing import List, Dict, Any
from prompts2 import build_prompt, LEVELS
# ============= 你可以在這裡調參 =============
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
MAX_CHARS = 60                # 每句改寫長度上限
RANDOM_SEED = 7
SLEEP_SEC_BETWEEN_CALLS = 0.02
# ===========================================

random.seed(RANDOM_SEED)

# -----------------------------
# 45 句「直述句」句子庫（互不相關）
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
]

# -----------------
# 模型呼叫（HF）
# -----------------
def _init_model(): 
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tok, mdl

def call_model(tokenizer, model, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        text = "User:\n" + prompt + "\nAssistant:\n"

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=1.5,#可改 越高似乎會越創意 控制隨機性，越高 → 越創意 / 越不確定
        top_p=0.9,#核心抽樣 (nucleus sampling)，保留累積機率 90% 的 token
        do_sample=True,#啟用隨機抽樣，否則會選擇機率最高的 token（greedy）
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    resp = decoded.splitlines()[-1].strip()
    return resp[:MAX_CHARS]

def generate_rows() -> List[Dict[str, Any]]:
    tok, mdl = _init_model()
    rows: List[Dict[str, Any]] = []
    total = len(SENTENCES)
    for i, s in enumerate(SENTENCES, start=1):
        print(f"[{i:>2}/{total}] 改寫：{s}")
        for lvl in LEVELS:
            prompt = build_prompt(s, lvl)
            resp = call_model(tok, mdl, prompt).strip()
            if not resp:
                continue
            rows.append({"sentence": s, "level": lvl, "response": resp})
            if SLEEP_SEC_BETWEEN_CALLS:
                time.sleep(SLEEP_SEC_BETWEEN_CALLS)
    return rows

def save_csv(rows: List[Dict[str, Any]], out_dir: str = "outputs") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "change_style.csv")

    # 依句子彙整（每句一行，含 Level1~Level5）
    grouped: Dict[str, Dict[int, str]] = {}
    for r in rows:
        s = r["sentence"]
        grouped.setdefault(s, {})[r["level"]] = r["response"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["原始句子"] + [f"Level{i} 改寫" for i in range(1, 6)])
        for s in SENTENCES:
            lv_map = grouped.get(s, {})
            row = [s] + [lv_map.get(i, "") for i in range(1, 6)]
            writer.writerow(row)

    print(f"✅ 已輸出：{csv_path}")
    return csv_path

if __name__ == "__main__":
    try:
        rows = generate_rows()
        save_csv(rows)
    except Exception as e:
        # 讓使用者知道可能是模型未安裝或無網路
        print("執行失敗：", repr(e))
        print("請確認已安裝 transformers、可下載模型，或改為本地可用的模型名稱。")
