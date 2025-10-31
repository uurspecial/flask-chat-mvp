
"""
Scenario Response Generator
---------------------------
用途：針對「大量不同情境」自動產生多語氣/多形式的回覆（不限定酸），
      供後續分類器評分或做資料擴增。本程式僅負責生成與存檔。

輸出：
  - ./outputs/scenario_responses.csv
  - ./outputs/scenario_responses.xlsx  (若環境有安裝 pandas)

欄位：
  - scenario_id, scenario, user_utterance, tone, style, response
"""

import os, csv, re, time, random
from typing import List, Dict, Any, Tuple
from pathlib import Path

# ============= 你可以在這裡調參 =============
# 模型（建議：Qwen2.5-7B-Instruct。顯存較小可用 Qwen2.5-1.5B-Instruct / phi-3-mini-128k-instruct）
MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

# 每個情境要產生幾種回覆
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
    # 🧍‍♀️ 人際互動
    "朋友臨時放你鴿子",
    "同事愛講風涼話",
    "室友總是不倒垃圾",
    "家人一直拿你和別人比較",
    "朋友只在需要幫忙時出現",
    # 🍽 日常生活
    "外送餐點送錯還怪客人",
    "餐廳菜太鹹但店員說是正常",
    "排隊被插隊卻沒人制止",
    "購物網站訂單延遲但客服只說稍等",
    "修好的手機比之前更壞",
    # 🏫 工作學業
    "組員報告完全不做事",
    "主管開會只會挑錯不給建議",
    "同學交白卷卻分數比你高",
    "老師突然改交作業時間",
    "面試官問跟能力無關的私事",
    # 💬 網路社交
    "群組已讀不回卻在別處聊天",
    "發文被人酸但對方說是開玩笑",
    "留言被曲解導致誤會",
    "別人截你對話去亂傳",
    "被陌生人私訊問奇怪問題",
    # 🚗 公共場合
    "公車司機亂罵學生沒投錢",
    "餐廳員工打翻飲料卻怪客人",
    "理髮師剪壞髮型還說很好看",
    "醫院排隊被插隊卻沒人處理",
    "銀行窗口愛理不理",
]


# =========================
# 2) 語氣/風格（不含酸度，側重多樣）
# =========================
TONES = [
    "中性／普通表述：語氣平穩、沒有情緒起伏，像是在描述事實或隨口回應。語句短、自然、無明顯態。",
    
    "輕微不滿／冷淡評論：語氣帶一點不耐、失望或無奈，但仍算克制。可出現輕微反諷或嘆息感，像是口頭碎念。",
    
    "明顯挖苦／挑釁語氣：語氣開始明確帶刺，有輕蔑或諷刺意味，但仍屬輕度對話。像是在“笑著酸人”。",
    
    "冷酸／高階反諷語氣：語氣表面冷靜、語言流暢，內容卻滿是諷刺。常以理性、假裝誇獎的方式表達反諷。像是高級的冷笑話。",
    
    "極端酸／高段反諷：語氣狠但有梗，聽起來像機智的狠話。帶強烈嘲諷或高智商式吐槽，內容犀利卻像在開玩笑。句子節奏強、反差感重。"
]




# =========================
# 3) Prompt 與模型呼叫
# =========================
def build_prompt(scenario: str, tone: str) -> str:
    """
    指定酸度層級生成對應語氣的回覆。
    """
    return f"""你是一個「語氣模擬生成器」，要模擬人類在不同情境下的回覆語氣。
請根據指定的「酸度層級」生成一句自然中文的回覆，讓語氣符合等級差異。

規則：
- Level 1 為最中性平淡；Level 5 為最酸、最帶刺。
- 請確保每一層的語氣差異明顯。
- 用自然中文，不要過度禮貌或機器化。
- 限制在 {MAX_CHARS} 字以內。
- 不要解釋、不要加評論，只輸出句子本體。
- 不得出現 AI、機器、系統、生成等字樣。
- 語氣必須像真實人講話。

情境：{scenario}
酸度層級：{tone}
"""


# === Hugging Face 模型初始化（整支程式只做一次） ===
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def call_model(prompt: str) -> str:
    """使用 Hugging Face Transformers 生成回覆"""
    messages = [{"role": "user", "content": prompt}]
    # 多數 Instruct/Chat 模型支援 chat template；若不支援可改成直接拼接字串
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
    # 嘗試取最後一段（避免模板殘留）
    resp = decoded.split("assistant")[-1].split("\n")[-1].strip()
    # 長度裁切
    return resp[:MAX_CHARS]

def normalize(txt: str) -> str:
    return re.sub(r"\s+", "", txt)

def generate() -> List[Dict[str, Any]]:
    """依據情境與酸度層級產生句子"""
    rows: List[Dict[str, Any]] = []
    total = len(SCENARIOS)

    for i, scenario in enumerate(SCENARIOS, start=1):
        scenario_id = f"S{i:03d}"

        # 只用 TONES，隨機打散後取前 N 個
        tone_pool = list(TONES)
        random.shuffle(tone_pool)
        take = tone_pool[:NUM_RESP_PER_SCENARIO]

        print(f"[{i:>3}/{total}] {scenario} -> 產生 {len(take)} 筆（tones）")
        for tone in take:
            # 新的 prompt 只吃 (scenario, tone)
            prompt = build_prompt(scenario, tone)
            resp = call_model(prompt).strip()
            if not resp:
                continue

            rows.append({
                "scenario_id": scenario_id,
                "scenario": scenario,
                "user_utterance": "",   # 如需也產生「用戶說法」，可再擴充
                "tone": tone,
                "response": resp[:MAX_CHARS],
            })

            if SLEEP_SEC_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_SEC_BETWEEN_CALLS)

    return rows

def save_outputs(rows: List[Dict[str, Any]], out_dir: str = "outputs") -> Tuple[str, str]:
    """只輸出 response 欄位為 CSV + TXT，並自動避免覆蓋"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 只取出回覆欄位
    responses = [r["response"] for r in rows if "response" in r]

    # 自動生成不重複檔名
    def next_available_path(base_name: str, ext: str) -> str:
        """自動避開覆蓋，生成 like only_responses_2.csv"""
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

    # 寫入 CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for resp in responses:
            writer.writerow([resp])

    # 寫入 TXT
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        for resp in responses:
            f.write(resp.strip() + "\n")

    print(f"✅ 已輸出：\n - {csv_path}\n - {txt_path}")
    return csv_path, txt_path


if __name__ == "__main__":
    rows = generate()
    if rows:
        csv_path, txt_path = save_outputs(rows)
    else:
        print("No rows generated.")
