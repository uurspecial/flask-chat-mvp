# -*- coding: utf-8 -*-
"""
seprompt_data.py
功能：讀取 dataset1.json (JSONL格式) 並作為 Dynamic Few-Shot 來源
Method: breeze_with_data (有資料訓練/參考)
"""
import json
import random
import os

# =========================
# 1. 定義語氣準則
# =========================
LEVEL_DEFS = {
    1: "【Level 1：溫和】(重點在對方感受，語氣柔和)",
    2: "【Level 2：中性】(陳述事實，冷靜直接)",
    3: "【Level 3：不滿】(明顯不耐煩，帶情緒壓力)",
    4: "【Level 4：酸】(反語、假稱讚、陰陽怪氣)"
}

LEVELS = [1, 2, 3, 4]

# =========================
# 2. 資料載入模組 (修正版：支援 JSON Lines)
# =========================
DATA_STORE = {1: [], 2: [], 3: [], 4: []}
DATA_LOADED = False

def load_dataset_from_json(json_path="dataset1.json"):
    """
    讀取 dataset1.jsonl (JSONL 格式) 並依照 score 分類存入 DATA_STORE
    """
    global DATA_LOADED
    
    # 防止重複讀取
    if DATA_LOADED:
        return

    if not os.path.exists(json_path):
        print(f"⚠️ 警告：找不到 {json_path}，將無法使用資料範例！")
        return

    try:
        count = 0
        with open(json_path, "r", encoding="utf-8") as f:
            # === 修改重點：逐行讀取 JSON Lines ===
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue # 跳過空行
                
                try:
                    item = json.loads(line) # 解析單行 JSON
                    
                    text = item.get("text", "")
                    # 確保 score 是整數
                    score_raw = item.get("score", 0)
                    score = int(score_raw)
                    
                    if score in DATA_STORE and text:
                        DATA_STORE[score].append(text)
                        count += 1
                        
                except json.JSONDecodeError:
                    print(f"⚠️ 跳過無法解析的行 (Line {line_number})")
                    continue
                except ValueError:
                    continue # score 轉換失敗跳過
        
        DATA_LOADED = True
        print(f"✅ 成功載入 {json_path}，共 {count} 筆資料。")
        # 顯示載入統計
        stats = ", ".join([f"L{k}:{len(v)}" for k,v in DATA_STORE.items()])
        print(f"   (統計: {stats})")
        
    except Exception as e:
        print(f"❌ 讀取檔案發生未預期錯誤: {e}")

# =========================
# 3. Prompt 建構器
# =========================
def build_prompt(original_text: str, level: int) -> str:
    """
    動態組裝 Prompt。
    從 DATA_STORE 中隨機抽取範例來構建 prompt。
    """
    if level not in LEVELS:
        return "Error: Level must be 1, 2, 3, or 4."

    # 確保資料已載入
    if not DATA_LOADED:
        load_dataset_from_json()

    target_desc = LEVEL_DEFS.get(level)
    level_labels = {1: "一級改寫", 2: "二級改寫", 3: "三級改寫", 4: "四級改寫"}
    current_label = level_labels[level]
    
    # === 從資料庫隨機撈取 3 句作為參考 ===
    n_shots = 30
    available_examples = DATA_STORE.get(level, [])
    
    if len(available_examples) > n_shots:
        selected_examples = random.sample(available_examples, n_shots)
    else:
        selected_examples = available_examples
    
    # 組合範例字串
    examples_str = ""
    if selected_examples:
        examples_str = "以下是此語氣的【參考範例句】(請學習其用詞與風格)：\n"
        for ex in selected_examples:
            examples_str += f"- {ex}\n"
    else:
        examples_str = "(此等級目前無參考資料，請依定義發揮)"

    return f"""你是一位語氣改寫專家。
請將「原句」改寫為「{current_label}」版本。

### 目標語氣：
{target_desc}

### 參考範例：
{examples_str}

### 你的任務：
原句："{original_text}"
{current_label}："""
