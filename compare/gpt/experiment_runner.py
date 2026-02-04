import pandas as pd
from openai import OpenAI
import sys
import io
import os
import random
from dotenv import load_dotenv

# --- [設定] 強制輸出編碼 & 載入 Key ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ 錯誤：找不到 OPENAI_API_KEY。")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# =========================
# 0. 載入外部資料集 (dataset1.jsonl)
# =========================
print("📂 正在載入 dataset1.jsonl ...")
try:
    df_dataset = pd.read_json("dataset1.jsonl", lines=True)
    df_dataset['score'] = df_dataset['score'].astype(int)
    print(f"✅ 資料集載入成功，共 {len(df_dataset)} 筆資料。")
except Exception as e:
    print(f"❌ 無法讀取 dataset1.jsonl: {e}")
    sys.exit(1)

def get_dataset_references(level, use_all=True):
    """
    從 dataset1.jsonl 撈取該等級的句子。
    use_all=True: 撈全部 (不限制數量)
    """
    try:
        filtered_df = df_dataset[df_dataset['score'] == level]
        if filtered_df.empty:
            return ""
        
        # 撈取全部資料
        samples = filtered_df['text'].tolist()
        
        # 為了避免 Prompt 太長超過 Token 限制，建議還是設個上限 (例如 50~100)
        # 如果你堅持要全部，請確保你的 Token 額度足夠
        # 這裡設為最多 50 句，若要真的全部請拿掉 [:50]
        formatted_samples = "\n".join([f"- {s}" for s in samples[:50]]) 
        return f"\n\n【額外參考風格庫 (Level {level})】：\n{formatted_samples}"
    except Exception as e:
        return ""

# =========================
# 1. 定義測試句子
# =========================
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

# =========================
# 2. 定義 Few-Shot (結構化範例)
# =========================
# 這裡定義的是「同一句話在不同等級」的對照，讓模型學習轉換邏輯
FEW_SHOT_DATA = [
    {
        "original": "你怎麼還沒來",
        1: "我有點擔心你，路上還順利嗎？慢慢來沒關係。",
        2: "路上塞車嗎，你已經遲到了喔。",
        3: "時間都過多久了，你卻還沒出現，大家都在等你你知道嗎?",
        4: "你的時間安排真的很自由，大家配合你就好。"
    },
    {
        "original": "為什麼不回我訊息",
        1: "剛剛沒看到你的回覆，我有點關心你是不是在忙。",
        2: "你還沒回我訊息，在幹嘛",
        3: "訊息放著不回，我等很久了，你到底在做甚麼。",
        4: "原來回訊息也要看心情，長知識了，呵呵。"
    },
    {
        "original": "你報告做完了嗎",
        1: "想關心一下你那份報告進行得怎麼樣，需要我陪你一起看看嗎？",
        2: "那份報告你現在做到哪了?",
        3: "報告拖成這樣，你如果不想做可以直接說。",
        4: "你這進度控制得真穩定，幾乎沒有變化。"
    },
    {
        "original": "你今天心情不好嗎",
        1: "我感覺你今天情緒有點不好，如果想要聊天，我在這裡。",
        2: "你今天看起來心情很差欸。",
        3: "你在不爽甚麼，莫名其妙就要挨你一頓氣。",
        4: "你今天情緒表現得很完整，大家都感受到了。"
    },
    {
        "original": "你剛剛說甚麼",
        1: "剛剛那段我沒聽清楚，可以請你再說一次嗎？",
        2: "我剛沒聽清楚，你再說一次。",
        3: "你講話又快小聲我真的聽不到，可不可以好好再說一次阿。",
        4: "你剛剛那段說明很有深度，沒人聽得懂呢。"
    }
]

# =========================
# 3. Prompt 生成函式
# =========================
def make_prompt(current_input, target_level, use_dataset=True):
    """
    current_input: 目前的輸入句子 (可能是原句，也可能是上一層的改寫結果)
    target_level: 目標等級 (1~4)
    """
    
    # 建構 Few-Shot 字串 (顯示目標等級的範例)
    # 策略：因為是階梯式，我們告訴模型：「不管你拿到什麼，請把它改成 Level X」
    examples_str = "【改寫示範】：\n"
    for item in FEW_SHOT_DATA:
        # 這裡展示：不管原句是什麼，最終 Level X 長這樣
        examples_str += f"原句：{item['original']} -> Level {target_level}：{item[target_level]}\n"

    # 撈取額外資料集 (針對目標等級)
    extra_data_str = ""
    if use_dataset:
        extra_data_str = get_dataset_references(target_level)
    
    instruction = f"請參考以下風格，將輸入句子改寫為「Level {target_level}」的語氣。\n注意：請直接輸出改寫後的句子，不要解釋。\n\n"
    
    prompt = f"{instruction}{examples_str}{extra_data_str}\n\n輸入句子：{current_input} -> 改寫："
    return prompt

# =========================
# 4. 主程式迴圈 (階梯式 + Long Format)
# =========================
results = []
METHODS = ["basic", "data"] # basic: 沒資料, data: 有資料
total = len(SENTENCES)

print(f"🔄 開始執行階梯式改寫實驗，共 {total} 句...")

for i, start_sentence in enumerate(SENTENCES, start=1):
    seq_id = f"S{i:03d}"
    print(f"[{i:>2}/{total}] ID: {seq_id} | 初始句子：{start_sentence}")

    # 針對兩種方法分別跑 (有資料 vs 沒資料)
    for method in METHODS:
        use_dataset = (method == "data")
        
        # ⭐ 重置：每一種方法的起點都是原始句子
        current_input = start_sentence
        
        # 階梯式迴圈：L1 -> L2 -> L3 -> L4
        for lvl in range(1, 5):
            try:
                prompt = make_prompt(current_input, lvl, use_dataset=use_dataset)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                resp_text = response.choices[0].message.content.strip()
                if not resp_text:
                    resp_text = "(生成失敗)"

                # 加入結果列表 (Long Format)
                results.append({
                    "id": seq_id,
                    "original": current_input,  # 上一層的輸出 = 這一層的輸入
                    "method": method,           # basic 或 data
                    "target_level": lvl,
                    "output": resp_text
                })
                
                # ⭐⭐ 關鍵更新：把目前的輸出變成下一層的輸入
                current_input = resp_text
                
            except Exception as e:
                print(f"⚠️ Error in {seq_id}, {method}, Lv{lvl}: {e}")
                results.append({
                    "id": seq_id,
                    "original": current_input,
                    "method": method,
                    "target_level": lvl,
                    "output": "Error"
                })
                # 發生錯誤時，輸入不更新，嘗試用舊的繼續跑 (或你可以選擇 break)
    
    print("    ✅ 完成")

# =========================
# 5. 存檔
# =========================
output_file = "experiment_new.csv"
df_result = pd.DataFrame(results)

# 指定欄位順序
cols = ["id", "original", "method", "target_level", "output"]
df_result = df_result[cols]

df_result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n🎉 實驗完畢！結果已儲存為 {output_file}")
print("內容預覽：")
print(df_result.head(10))
