# -*- coding: utf-8 -*-
"""
讀取問題庫 -> 呼叫 reprompt.py 產生 Prompt -> 讓模型生成 -> 存成 JSONL 格式
改動回覆的樣子 level字數
"""

import os, csv, time, random, json
from typing import List, Dict, Any
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# 1) 引用剛寫好的 Prompt 模組
# =========================
try:
    from rprompt1 import build_prompt, LEVELS
except ImportError:
    print("❌ 錯誤：找不到 reprompt.py！請確認檔案名稱是否正確。")
    exit()

# =========================
# 2) 參數設定
# =========================
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
NUM_RESP_PER_QUESTIONS = 4   # 產生 Level 1~4
MAX_CHARS = 60               # 限制生成長度
SLEEP_SEC_BETWEEN_CALLS = 0.1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# 3) 問題庫 (Scenario)
# =========================
QUESTIONS = [
    "你今天有空一起吃個飯嗎？",
    "外送送錯了，你覺得我要跟店家說嗎？",
    "你覺得這件衣服適合我嗎？",
    "你最近有推薦的餐廳嗎？",
    "為什麼我手機最近一直在當？",
    "朋友臨時取消約，我該怎麼回比較好？",
    "如果別人一直已讀不回，是不是代表他不想回我？",
    "室友常常不倒垃圾，我要怎麼講才不會吵架？",
    "家人一直拿我跟別人比較，我該怎麼面對？",
    "老師突然改期限，我還能說什麼嗎？",
    "如果組員都不做事，我該怎麼辦？",
    "主管一直挑錯但不給建議，我要怎麼應對？",
    "我覺得薪水有點低，你覺得我該怎麼開口？",
    "面試官問一些很奇怪的問題，我該怎麼回答？",
    "網購延遲這麼久都沒到，我該怎麼處理？",
    "客服一直叫我稍等但都沒更新，你覺得該怎麼辦？",
    "餐廳上錯菜但店員說是我點錯的，我該怎麼回？",
    "覺得髮型剪壞了，你會直接跟設計師講嗎？",
    "保修回來的東西變更壞了，我該怎麼說？",
    "排隊有人插隊，我要不要講？",
    "公車司機態度很差，你會回嘴嗎？",
    "醫院排隊很久卻有人插隊，我該怎麼反應？",
    "銀行櫃台態度很冷淡，我該怎麼講比較好？",
    "如果陌生人突然私訊我，我要回嗎？",
    "朋友只在需要幫忙時才找我，我該怎麼回他？",
    "同事講話很酸，我要怎麼回才不會吵架？",
    "別人問我一些很私人的問題，我應該怎麼回答？",
    "如果對方一直催我做事，我該怎麼回比較得體？",
    "有人把我的訊息截圖傳來傳去，我該怎麼回應？",
    "發文被酸但是他說是在開玩笑，我要怎麼反應？"
]

# =========================
# 4) 初始化模型
# =========================
print(f"🔄 正在載入模型：{MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("✅ 模型載入完成！")

def call_model(prompt: str) -> str:
    """呼叫模型並回傳生成的文字"""
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return resp[:MAX_CHARS]

def generate() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(QUESTIONS)

    print("🚀 開始生成...")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i:>2}/{total}] 處理問題：{question}")

        for level in LEVELS[:NUM_RESP_PER_QUESTIONS]:
            prompt = build_prompt(question, level, max_chars=MAX_CHARS)
            resp = call_model(prompt)
            if not resp: resp = "(生成失敗)"

            # 存入列表
            rows.append({
                "response": resp,
                "level": level
            })
            
            # 在終端機印出來看一下
            print(f"    -> Lv{level}: {resp}")

            if SLEEP_SEC_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_SEC_BETWEEN_CALLS)

    return rows

def save_outputs(rows: List[Dict[str, Any]], out_dir: str = "outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 設定輸出檔名
    jsonl_path = os.path.join(out_dir, "reply.jsonl")

    # 開始寫入 JSONL 格式
    print(f"\n💾 正在儲存到 {jsonl_path} ...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            # 建立你要的格式物件
            data_line = {
                "text": row["response"],
                "score": str(row["level"])  # 將數字轉成字串 "1", "2"...
            }
            # 轉成 json 字串並寫入一行 (ensure_ascii=False 確保中文不會變亂碼)
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")

    print(f"✅ 生成完畢！")

if __name__ == "__main__":
    generated_data = generate()
    if generated_data:
        save_outputs(generated_data)
    else:
        print("⚠️ 沒有生成任何資料。")