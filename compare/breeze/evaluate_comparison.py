# -*- coding: utf-8 -*-
"""
evaluate_comparison.py
功能：
1. 讀取 run_comparison.py 產生的 comparison_output.csv。
2. 使用 OpenAI API 對 'output' 欄位進行評分 (1-4)。
3. 比較 'predicㄌted_score' 與 'target_level'，計算準確率。
4. 輸出 breeze_prompt vs breeze_with_data 的勝負數據。
"""

import pandas as pd
import os
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# --- 初始化 OpenAI ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # 如果沒讀到，您也可以在這裡直接貼上 key (不建議上傳)
    # api_key = "sk-..." 
    print("⚠️ 警告：未設定 OPENAI_API_KEY，請檢查 .env")

client = OpenAI(api_key=api_key)

# --- 定義評分標準 (Prompt) ---
SCORING_PROMPT = """
請根據下面的語氣標準，給句子打分數（1～4）：

1-溫和（Polite/Warm）：語氣柔軟，常使用「不好意思、謝謝、麻煩你」，維護關係。
2-中性（Neutral/Factual）：像機器人或新聞報導，無情緒，僅傳遞資訊。
3-不滿（Direct Anger）：情緒直接外露，直接表達憤怒、指責、命令。
4-酸（Sarcastic）：陰陽怪氣、反諷、誇獎形式貶低、挖苦。

請閱讀句子並給出一個數字 (1, 2, 3, 或 4)。
只回傳數字即可。

句子：「{sentence}」
分數："""

def score_sentence_llm(sentence: str) -> int:
    """呼叫 GPT-4o-mini 評分"""
    if pd.isna(sentence) or str(sentence).strip() == "" or "(生成失敗)" in str(sentence):
        return -1

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": SCORING_PROMPT.format(sentence=sentence)}],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        
        # 解析數字
        match = re.search(r'\d+', content)
        if match:
            return int(match.group())
        return -1
        
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(1) # 簡單的 Rate limit 處理
        return -1

def main():
    input_csv = "comparison_output.csv"
    output_scored_csv = "comparison_scored_final.csv"

    if not os.path.exists(input_csv):
        print(f"❌ 找不到 {input_csv}，請先執行 run_comparison.py")
        return

    print(f"📖 讀取資料: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"🚀 開始評分 {len(df)} 筆資料...")
    
    # 使用 tqdm 顯示進度條
    tqdm.pandas(desc="評分進度")
    df['predicted_score'] = df['output'].progress_apply(score_sentence_llm)
    
    # --- 計算準確率 ---
    # 邏輯：模型預測的分數 (predicted_score) 是否等於 目標等級 (target_level)
    df['is_correct'] = (df['predicted_score'] == df['target_level'])
    
    # 儲存詳細結果
    df.to_csv(output_scored_csv, index=False, encoding="utf-8-sig")
    print(f"\n詳細評分結果已儲存至: {output_scored_csv}")

    # --- 產出比較報表 ---
    print("\n" + "="*40)
    print("實驗結果分析")
    print("="*40)

    # 1. 過濾掉評分失敗 (-1) 的資料
    valid_df = df[df['predicted_score'] != -1]
    print(f"有效樣本數: {len(valid_df)} / {len(df)}")

    # 2. 整體準確率比較 (Method vs Method)
    print("\n🏆 整體準確率 (Accuracy by Method):")
    accuracy_by_method = valid_df.groupby('method')['is_correct'].mean() * 100
    print(accuracy_by_method.round(2).astype(str) + '%')

    # 3. 各等級準確率比較 (Level details)
    print("\n各等級表現 (Accuracy by Method & Level):")
    pivot_table = valid_df.pivot_table(
        index='target_level', 
        columns='method', 
        values='is_correct', 
        aggfunc='mean'
    ) * 100
    print(pivot_table.round(2).astype(str) + '%')

if __name__ == "__main__":
    main()
