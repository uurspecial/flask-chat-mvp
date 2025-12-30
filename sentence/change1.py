# -*- coding: utf-8 -*-
"""
1vs4sentence.py
---------------------------------
1) 讀取句子庫 (SENTENCES)
2) 呼叫 seprompt3.py 取得 Prompt (含 Few-Shot 範例)
3) 產出 outputs/1vs4sentence_chain.csv
   格式：原始句子 | Level1 改寫 | Level2 改寫 | Level3 改寫 | Level4 改寫

【改動重點】
- 原本：每個 level 都用原始句子 s 生成（平行）
- 現在：Level1 -> Level2 -> Level3 -> Level4（階梯式）
- 新增：sanitize_resp()，避免模型輸出「Level X 改寫：...」這種標頭污染下一層輸入
"""

import os, csv, time, random, re
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# 1. 引用 Prompt 模組
# =========================
try:
    from seprompt3 import build_prompt, LEVELS
except ImportError:
    print("❌ 錯誤：找不到 seprompt3.py！")
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
SENTENCES: list[str] = [
    "今天外面好冷喔",
    "我剛醒來腦袋還沒轉過來",
    "這首歌好好聽阿阿阿",
    "我突然有點餓",
    "你剛剛那句話很好笑",
    "我現在不想說話",
    "這地方我以前好像來過",
    "今天一整天都覺得很累",
    "我突然想到一件小事",
    "這個味道讓我有點反胃",
    "你這個反應也太誇張了吧",
    "我其實蠻喜歡這種天氣的",
    "現在這個時間點有點尷尬",
    "我有點想睡又睡不著",
    "這樣看起來好像滿順眼的",
    "你不覺得這樣很奇怪嗎",
    "我剛剛差點忘記帶東西",
    "這句話聽起來有點熟",
    "我現在只想放空一下",
    "你怎麼突然講這個",
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


# -----------------
# 輸出清洗：避免標頭/多行污染
# -----------------
_BAD_PREFIX = re.compile(
    r"^\s*(?:Level\s*\d+\s*改寫|Lv\.?\s*\d+|原句|改寫)\s*[:：]\s*",
    flags=re.IGNORECASE
)

def sanitize_resp(resp: str, max_chars: int = MAX_CHARS) -> str:
    """
    1) 只取第一行（避免模型吐多行/段落）
    2) 去掉常見標頭：Level X 改寫： / 原句： / 改寫：
    3) 去掉外層引號
    4) 截斷到 max_chars
    """
    if not resp:
        return resp

    # 只取第一行
    resp = resp.strip().splitlines()[0].strip()

    # 去標頭
    resp = _BAD_PREFIX.sub("", resp).strip()

    # 去外層引號
    resp = resp.strip(' "\'“”‘’`')

    # 長度上限
    return resp[:max_chars]


def call_model(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    # 套用 Chat Template
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,     # 0.8 較穩，過高容易亂飛
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # --- 精準截取回覆（只截新增 tokens）---
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 不在這裡截 MAX_CHARS（避免把標頭截一半造成更亂）
    return resp


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

            raw = call_model(tokenizer, model, prompt)
            resp = sanitize_resp(raw, MAX_CHARS)

            if not resp:
                resp = "(生成失敗)"

            rows.append({
                "sentence": s,      # 保留最原始句子
                "level": lvl,
                "response": resp
            })

            print(f"    -> Lv{lvl}: {resp}")

            # ⭐⭐ 關鍵：把清洗後的輸出當下一層輸入（避免 Level 標頭污染）
            current_sentence = resp

            if SLEEP_SEC_BETWEEN_CALLS:
                time.sleep(SLEEP_SEC_BETWEEN_CALLS)

    return rows


def save_csv(rows: List[Dict[str, Any]], out_dir: str = "outputs") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def get_next_csv_path(out_dir: str, base_name: str) -> str:
        # 第 1 個：base_name.csv
        first = os.path.join(out_dir, f"{base_name}.csv")
        if not os.path.exists(first):
            return first

        # 後面：base_name2.csv、base_name3.csv...
        idx = 2
        while True:
            path = os.path.join(out_dir, f"{base_name}{idx}.csv")
            if not os.path.exists(path):
                return path
            idx += 1

    csv_path = get_next_csv_path(out_dir, "1vs4sentence")

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
