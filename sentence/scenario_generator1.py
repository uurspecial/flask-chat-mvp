
# -*- coding: utf-8 -*-
# 這份檔案是對 scenario_generator1.py 的「最小幅度修改」版本：
# 1) 將五級準則搬到 prompts.py
# 2) 主程式只需「情境 + 等級」→ build_prompt() → call_model()
# 3) 每個情境預設輸出五句（對應 Level 1~5）

import os, csv, re, time, random
from typing import List, Dict, Any, Tuple
from pathlib import Path

# ============= 你可以在這裡調參 =============
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

# 每個情境要產生幾種回覆（預設為 5 對應 Level 1~5）
NUM_RESP_PER_SCENARIO = 5

# 句子最長字數（避免太長）
MAX_CHARS = 60

# 生成隨機種子（可重現）
RANDOM_SEED = 7

# 是否在大量生成時小睡一下（避免顯卡/CPU 滿載）
SLEEP_SEC_BETWEEN_CALLS = 0.02
# ===========================================

try:
    import pandas as pd
except Exception:
    pd = None

random.seed(RANDOM_SEED)

# =========================
# 1) 情境庫（可持續擴充）
# =========================
SCENARIOS = [
    "朋友臨時放你鴿子",
    "同事愛講風涼話",
    "室友總是不倒垃圾",
    "家人一直拿你和別人比較",
    "朋友只在需要幫忙時出現",
]

# =========================
# 2) 準則由 prompts.py 提供
# =========================
from prompts1 import build_prompt, LEVELS

# =========================
# 3) 模型呼叫
# =========================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def call_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "User:\n" + prompt + "\nAssistant:\n"

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resp = decoded.split("assistant")[-1].split("\n")[-1].strip()
    return resp[:MAX_CHARS]

def generate() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(SCENARIOS)

    for i, scenario in enumerate(SCENARIOS, start=1):
        scenario_id = f"S{i:03d}"
        print(f"[{i:>3}/{total}] {scenario} -> 產生 {len(LEVELS)} 筆（Level 1~5）")

        for level in LEVELS[:NUM_RESP_PER_SCENARIO]:
            prompt = build_prompt(scenario, level, max_chars=MAX_CHARS)
            resp = call_model(prompt).strip()
            if not resp:
                continue

            rows.append({
                "scenario_id": scenario_id,
                "scenario": scenario,
                "level": level,
                "response": resp,
            })

        if SLEEP_SEC_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_SEC_BETWEEN_CALLS)

    return rows

def save_outputs(rows: List[Dict[str, Any]], out_dir: str = "outputs") -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def next_available_path(base_name: str, ext: str) -> str:
        full_path = os.path.join(out_dir, f"{base_name}{ext}")
        if not os.path.exists(full_path):
            return full_path
        counter = 1
        while True:
            new_path = os.path.join(out_dir, f"{base_name}_{counter}{ext}")
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    csv_path = next_available_path("only_responses", ".csv")
    txt_path = next_available_path("only_responses", ".txt")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([r["response"]])

    with open(txt_path, "w", encoding="utf-8-sig") as f:
        for r in rows:
            f.write(r["response"] + "\n")

    print(f"✅ 已輸出：\n - {csv_path}\n - {txt_path}")
    return csv_path, txt_path


if __name__ == "__main__":
    rows = generate()
    if rows:
        save_outputs(rows)
    else:
        print("No rows generated.")
