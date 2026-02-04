import pandas as pd
import sys
import io
import os
import time

# --- 1. 匯入你的 Scorer (score_sentences.py) ---
# 確保 score_sentences.py 在同一個目錄下
try:
    # 修正：確認檔名是單數 score_sentences
    import score_sentences as scorer
except ImportError:
    print("❌ 錯誤：找不到 score_sentences.py。請確保該檔案在同一個目錄下。")
    sys.exit(1)

# --- 設定標準輸出編碼 ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 輸入與輸出檔案
INPUT_FILE = "experiment_new.csv"      # 包含所有選手產出結果的 CSV
OUTPUT_FILE = "final_scored_report.csv" # 評分後的結果

def main():
    # 1. 檢查輸入檔案
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 錯誤：找不到 {INPUT_FILE}。請先執行 experiment_runner.py 產生此檔案。")
        return

    # 2. 讀取比賽結果
    print(f"📂 正在讀取 {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    # --- [修正 1] 拆解型別轉換步驟，避免 float 錯誤 ---
    # 先轉為數字 (無法轉的變 NaN)
    df['target_level'] = pd.to_numeric(df['target_level'], errors='coerce')
    # 再填補 NaN 為 0 並轉為整數
    df['target_level'] = df['target_level'].fillna(0).astype(int)

    # 3. 準備評分
    total_rows = len(df)
    print(f"🚀 開始評分，共 {total_rows} 筆資料...")
    print("裁判：GPT-4o-mini (基於 dataset1.jsonl 的 Few-Shot 標準)")

    # --- 重置 Scorer 的進度計數器 ---
    scorer.progress_counter = 0
    scorer.total_sentences = total_rows

    predicted_scores = []
    is_correct_list = []

    # 4. 執行評分迴圈
    # --- [修正 2] 使用 enumerate 取得整數計數器 i，忽略原本的 index (可能是 S001 字串) ---
    for i, (index, row) in enumerate(df.iterrows()):
        
        output_text = str(row['output']).strip()
        target = row['target_level']
        
        # 如果生成結果是 Error 或 空值，直接判錯，不浪費 API
        if output_text == "Error" or output_text == "(生成失敗)" or not output_text:
            score = -1
            # 使用 i + 1 來顯示進度，確保是數字
            print(f"[{i+1}/{total_rows}] ⚠️ 跳過無效輸出 -> -1")
            scorer.progress_counter += 1
        else:
            # ⭐ 呼叫 Scorer ⭐
            score = scorer.score_sentence(output_text)
        
        # 判斷是否正確
        is_correct = (score == target)
        
        predicted_scores.append(score)
        is_correct_list.append(is_correct)

    # 5. 將結果寫回 DataFrame
    df['predicted_score'] = predicted_scores
    df['is_correct'] = is_correct_list

    # 6. 存檔
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 評分完成！詳細結果已存至 {OUTPUT_FILE}")

    # =========================
    # 7. 產生分析報表
    # =========================
    print("\n" + "="*40)
    print("🏆 最終對決結果 (準確率排行榜)")
    print("="*40)

    if 'method' in df.columns:
        # 依方法分組計算準確率
        leaderboard = df.groupby('method')['is_correct'].mean().sort_values(ascending=False) * 100
        
        print(f"{'Method (方法)':<20} | {'Accuracy (準確率)':<15}")
        print("-" * 40)
        for method, acc in leaderboard.items():
            print(f"{method:<20} | {acc:.2f}%")
        
        print("\n" + "="*40)
        print("📊 各等級詳細表現 (Level-wise Accuracy)")
        print("="*40)
        
        # 樞紐分析表：Method vs Level
        pivot = df.pivot_table(index='method', columns='target_level', values='is_correct', aggfunc='mean') * 100
        print(pivot.round(2))
        
        # 特別檢查 Breeze L4
        print("\n🔍 重點關注：Breeze 在 Level 4 (酸) 的表現")
        try:
            # 使用 contains 模糊搜尋 'breeze'
            breeze_l4 = pivot.loc[pivot.index.str.contains('breeze', case=False), 4]
            if not breeze_l4.empty:
                print(breeze_l4)
            else:
                print("找不到 Breeze 的 L4 資料")
        except:
            print("資料不足，無法分析 Breeze L4")

    else:
        print("⚠️ CSV 中找不到 'method' 欄位，無法產生分組比較表。")
        print(f"總體準確率: {df['is_correct'].mean() * 100:.2f}%")

if __name__ == "__main__":
    main()
