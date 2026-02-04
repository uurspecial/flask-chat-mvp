import pandas as pd
import json
from sklearn.model_selection import train_test_split
import re  
from openai import OpenAI 
from dotenv import load_dotenv 
import os 
import time # 🚀 新增: 匯入 time 模組，用於加入 API 呼叫之間的延遲

# --- 🎯 初始化區塊 (保持不變) 🎯 ---

# 載入 .env 檔
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ 錯誤：OPENAI_API_KEY 環境變數未設定。請檢查 .env 檔案。")

# 初始化 OpenAI 客戶端，使其成為全域變數 (Global client)
client = OpenAI(api_key=api_key)

# --- 🎯 區塊結束 🎯 ---
# === 步驟 1: 載入資料集 ===
def load_dataset_from_jsonl(file_path):
    """從 JSON Lines 檔案中載入資料集並轉換為 DataFrame。"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
        df['score'] = df['score'].astype(int) 
        return df
    except Exception as e:
        print(f"❌ 錯誤：載入或處理 {file_path} 時出錯: {e}")
        return None

# 載入所有資料
df_all = load_dataset_from_jsonl("dataset1.jsonl")

df_few_shot_examples = None
df_test = None

if df_all is not None:
    # === 步驟 2: 執行等比例 (Stratified) 訓練/測試分割 ===
    
    # 檢查是否有足夠的樣本進行分層抽樣（每個類別至少要有 2 個樣本）
    if min(df_all['score'].value_counts()) < 2:
        print("⚠️ 警告：某些類別的樣本數過少，無法執行分層抽樣，將使用一般隨機抽樣。")
        stratify_param = None
    else:
        stratify_param = df_all['score']
    
    df_few_shot_examples, df_test = train_test_split(
        df_all,
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_param # 使用分層參數
    )

    # 🚀 解除註解: 印出資料集分割結果
    print("\n📊 資料集分割結果:")
    print(f"Few-Shot 範例集大小: {len(df_few_shot_examples)}")
    print(df_few_shot_examples['score'].value_counts().sort_index())
    print("-" * 20)
    print(f"測試集大小: {len(df_test)}")
    print(df_test['score'].value_counts().sort_index())


scale_definition_new = """
請根據下面的語氣標準，給每一句話打分數（1～4）：

1-溫和（Polite / Warm）：語氣柔軟，常使用「不好意思、謝謝、麻煩你、喔、吧」等語助詞，目的是維護關係或展現禮貌。
2-中性（Neutral / Factual）：像機器人或新聞報導一樣陳述事實。不帶個人情緒，沒有明顯的語助詞，僅傳遞資訊。
3-不滿（Direct Anger / Complaint）：情緒直接外露。直接表達憤怒、指責、命令或抱怨。特徵是「直球對決」，不拐彎抹角，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
4-酸（Sarcastic / Mocking）：陰陽怪氣、高級反諷。使用「誇獎的形式來貶低」或「誇飾的比喻」。特徵是帶有幽默感、嘲諷、挖苦，比直接罵更刺耳。（例如：你的智商真是人類奇蹟）。
"""

# === 建立 Few-Shot 提示 (必須在 df_few_shot_examples 存在時才執行) ===
few_shot_prompt = scale_definition_new + "\n\n範例：\n"

# 確保 df_few_shot_examples 已被定義
if df_few_shot_examples is not None:
    for index, row in df_few_shot_examples.iterrows():
        few_shot_prompt += f"「{row['text']}」 → {row['score']}\n"
else:
    # 處理 df_all 為 None 的情況
    few_shot_prompt += "(未能載入範例資料)\n"


# === 步驟 4: 修正後的 score_sentence 函式 (加入進度印出和 global 簡化) ===
# 🚀 變數用於追蹤進度
progress_counter = 0 
total_sentences = len(df_test) if df_test is not None else 0

def score_sentence(sentence: str) -> int:
    """呼叫 LLM，對一句話打分數，包含自動 Retry 機制。"""
    global client, few_shot_prompt, progress_counter, total_sentences
    
    progress_counter += 1
    prompt = few_shot_prompt + f"\n請對下面句子打分：\n「{sentence}」 →"

    max_retries = 5  # 最多重試 5 次
    base_delay = 2   # 基礎等待時間 2 秒

    for attempt in range(max_retries):
        try:
            # 只有在第一次嘗試時印出進度，重試時印出重試訊息
            if attempt == 0:
                print(f"[{progress_counter}/{total_sentences}] 正在評分句子: 「{sentence[:10]}...」", end="")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            # --- 解析回應區塊 ---
            raw_score_content = response.choices[0].message.content
            if raw_score_content is None:
                print(" -> ❌ LLM 回應內容為 None")
                return -2
                
            raw_score = raw_score_content.strip()
            try:
                score = int(raw_score)
                print(f" -> ✅ 分數: {score}")
            except:
                numbers = re.findall(r"\d+", raw_score)
                score = int(numbers[0]) if numbers else -1
                print(f" -> ⚠️ 解析失敗，返回分數: {score}")
            
            # 成功後稍微休息一下，避免連續衝擊
            time.sleep(1) 
            return score

        except Exception as e:
            # 檢查是否為 Rate Limit Error (包含錯誤訊息中的關鍵字)
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                wait_time = base_delay * (2 ** attempt) # 指數退避: 2s, 4s, 8s, 16s...
                print(f"\n⚠️ 觸發速率限制 (429)，等待 {wait_time} 秒後重試... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # 其他錯誤直接報錯並結束
                print(f"\n⚠️ API 呼叫失敗。錯誤: {e}")
                return -3

    # 如果重試 5 次都失敗
    print(f"\n❌ 重試次數耗盡，放棄此句。")
    return -3

# === 主程式執行部分 ===
def main():
    global df_test # 確保能存取 df_test

    if df_all is None:
        print("\n🚨 由於資料載入失敗，程式終止。")
        return
    
    if df_test is None:
        print("\n🚨 由於資料分割失敗，程式終止。")
        return
    
    print("\n🚀 開始對測試集 (Test Set) 進行評分...")
    
    # 建立一個新的 'predicted_score' 欄位來存儲 LLM 的預測分數
    # 檢查 'text' 欄位是否存在
    if "text" not in df_test.columns:
         print("❌ 錯誤：測試集中找不到 'text' 欄位。")
         return
        
    df_test["predicted_score"] = df_test["text"].apply(score_sentence)
    
    print("\n✅ 評分完成。")

    # 存檔 (將測試集的結果存檔)
    OUTPUT_FILE = "scored_test_set_output.csv"
    df_test.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ 完成！結果已存到 {OUTPUT_FILE}")
    
    # 簡短報告
    accuracy = (df_test['score'] == df_test['predicted_score']).mean() * 100
    print(f"\n統計結果 (僅針對有效分數):")
    print(f"預測準確度: {accuracy:.2f}%")
    print(df_test['predicted_score'].value_counts().sort_index())

# 🚀 取消註解: 確保程式啟動時執行 main 函式
if __name__ == "__main__":
    main()
