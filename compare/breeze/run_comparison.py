# -*- coding: utf-8 -*-
"""
run_comparison.py
功能：
1. 同時執行 seprompt.py (breeze_prompt) 與 sepromptd.py (breeze_with_data)。
2. 針對同一組測試句，生成 L1->L4 的階梯式改寫。
3. 輸出 comparison_output.csv，格式對齊您的圖片要求。
"""

import os, csv, time, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# 1. 引用兩個 Prompt 模組
# =========================
try:
    import seprompt4   # 對應 Method: breeze_prompt (純 Few-Shot)
    import sepromptd  # 對應 Method: breeze_with_data (Dataset Few-Shot)
except ImportError as e:
    print(f"❌ 錯誤：找不到模組 ({e})。請確保 seprompt.py 和 sepromptd.py 在同目錄。")
    exit()

# ============= 參數設定 =============
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
OUTPUT_FILE = "comparison_output.csv"
MAX_CHARS = 80
RANDOM_SEED = 42
DATASET_PATH = "dataset1.jsonl" # 確保 sepromptd 讀取的到
# ===================================

random.seed(RANDOM_SEED)

# 測試句子
SENTENCES = [
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
    # 您可以自行增加更多句子
]

# -----------------
# 模型初始化
# -----------------
def _init_model():
    print(f"🔄 正在載入模型：{MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

# 清洗輸出的正規表達式
_BAD_PREFIX = re.compile(r"^\s*(?:Level\s*\d+\s*改寫|Lv\.?\s*\d+|[一二三四]級改寫|原句|改寫)\s*[:：]\s*", flags=re.IGNORECASE)

def sanitize_resp(resp: str) -> str:
    if not resp: return ""
    resp = resp.strip().splitlines()[0].strip()
    resp = _BAD_PREFIX.sub("", resp).strip()
    resp = resp.strip(' "\'“”‘’`')
    return resp[:MAX_CHARS]

def call_model(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.5, 
            top_p=0.9, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# -----------------
# 主執行邏輯
# -----------------
def run_generation():
    # 1. 預先載入 sepromptd 的資料 (如果有的話)
    print(" 初始化 Dataset...")
    sepromptd.load_dataset_from_json(DATASET_PATH)

    # 2. 載入模型
    tokenizer, model = _init_model()
    
    # module 從另外兩個叫進來
    methods = [
        {"name": "breeze_prompt",    "module": seprompt4},  # 1vs4 (純 Prompt)
        {"name": "breeze_with_data", "module": sepromptd}  # 1vs4d (含 Data)
    ]
    
    all_rows = []
    levels = [1, 2, 3, 4]

    print(f"🚀 開始執行比較實驗，共 {len(SENTENCES)} 句 x {len(methods)} 種方法...")

    for i, start_sentence in enumerate(SENTENCES, start=1):
        seq_id = f"S{i:03d}"
        print(f"\n[{i}/{len(SENTENCES)}] ID: {seq_id} | 原始: {start_sentence}")
        
        # 針對每一種方法，分別跑一次 L1->L4
        for method in methods:
            method_name = method["name"]
            mod = method["module"]
            
            # 重置輸入：每個方法的 Level 1 都是從最原始句子開始
            current_input = start_sentence
            
            # print(f"  👉 Method: {method_name}")
            
            for lvl in levels:
                # 呼叫各自模組的 build_prompt
                prompt = mod.build_prompt(current_input, lvl)

                # 生成回應
                raw_resp = call_model(tokenizer, model, prompt)
                resp = sanitize_resp(raw_resp)
                
                if not resp: resp = "(生成失敗)"
                
                # 收集資料 (格式對齊您的圖片)
                all_rows.append({
                    "id": seq_id,
                    "original": current_input,  # 這是「輸入給該 Level」的句子
                    "method": method_name,
                    "target_level": lvl,
                    "output": resp
                })
                
                # Chain Logic: 這一級的輸出 = 下一級的輸入
                current_input = resp

    # 4. 存檔
    print(f"\n💾 正在寫入結果：{OUTPUT_FILE}")
    fieldnames = ["id", "original", "method", "target_level", "output"]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"✅ 完成！檔案已儲存為 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_generation()
