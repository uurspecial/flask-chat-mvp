# scorer.py
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import re  
from openai import OpenAI 
import os 
import time

class Scorer:
    """
    將原有的 openai 評分邏輯封裝成 Class，方便主程式呼叫。
    """
    # 這裡加入 *args, **kwargs 是為了相容主程式傳入的 (api_key, model_name) 參數而不報錯
    def __init__(self, *args, dataset_path="dataset_scorer.jsonl", few_shot_n=50, **kwargs):
        # 1. 載入 OpenAI API Key (確保你的 .env 裡面有 OPENAI_API_KEY)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("❌ 錯誤：找不到 OPENAI_API_KEY 環境變數。請檢查 .env 檔案。")
        
        self.client = OpenAI(api_key=openai_key)
        self.model_name = "gpt-4o-mini"
        self.dataset_path = dataset_path
        self.few_shot_n = few_shot_n
        
        # 2. 語氣標準定義 (完全保留)
        self.scale_definition_new = """
請根據下面的語氣標準，給每一句話打分數（1～4）：

1-溫和（Polite / Warm）：語氣柔軟，常使用「不好意思、謝謝、麻煩你、喔、吧」等語助詞，目的是維護關係或展現禮貌。
2-中性（Neutral / Factual）：像機器人或新聞報導一樣陳述事實。不帶個人情緒，沒有明顯的語助詞，僅傳遞資訊。
3-不滿（Direct Anger / Complaint）：情緒直接外露。直接表達憤怒、指責、命令或抱怨。特徵是「直球對決」，不拐彎抹角，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
4-酸（Sarcastic / Mocking）：陰陽怪氣、高級反諷。使用「誇獎的形式來貶低」或「誇飾的比喻」。特徵是帶有幽默感、嘲諷、挖苦，比直接罵更刺耳。（例如：你的智商真是人類奇蹟）。
"""
        # 3. 初始化時就建立好 Few-Shot Prompt，避免每次打分都重新讀取檔案
        self.few_shot_prompt = self._build_few_shot_prompt(dataset_path)

    def _build_few_shot_prompt(self, file_path):
        """
        從 dataset_scorer.jsonl 讀取資料並建立 Few-Shot 範例
        - 隨機抽取 self.few_shot_n 筆資料作為示例
        - 優先嘗試分層抽樣，避免某一等級過少
        """
        data = []
        dataset_path = file_path or self.dataset_path
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 避免空行報錯
                        data.append(json.loads(line.strip()))
            
            # 轉換為 DataFrame 便於操作
            df_all = pd.DataFrame(data)
            if 'score' not in df_all.columns:
                raise ValueError("❌ dataset_scorer.jsonl 缺少 'score' 欄位")
            
            df_all['score'] = df_all['score'].astype(int)
            
            print(f"✅ 成功載入 {dataset_path}，共 {len(df_all)} 筆資料")
            stats = ", ".join([f"L{k}:{(df_all['score']==k).sum()}" for k in sorted(df_all['score'].unique())])
            print(f"   (統計: {stats})")
            
            # 隨機抽樣 self.few_shot_n 筆資料
            n_samples = min(self.few_shot_n, len(df_all))
            if n_samples < len(df_all):
                # 嘗試分層抽樣，若分層無法成立則隨機抽樣
                if df_all['score'].value_counts().min() >= 2:
                    df_sample = df_all.groupby('score', group_keys=False).apply(
                        lambda g: g.sample(n=max(1, int(round(n_samples * len(g) / len(df_all)))), random_state=42)
                    )
                    if len(df_sample) > n_samples:
                        df_sample = df_sample.sample(n=n_samples, random_state=42)
                else:
                    df_sample = df_all.sample(n=n_samples, random_state=42)
            else:
                df_sample = df_all
            
            df_few_shot = df_sample
        except FileNotFoundError:
            print(f"❌ 找不到 {dataset_path}，請確保檔案存在")
            return self.scale_definition_new + "\n\n範例：\n(未能載入範例資料)\n"
        except Exception as e:
            print(f"❌ 載入 {dataset_path} 出錯: {e}")
            return self.scale_definition_new + "\n\n範例：\n(未能載入範例資料)\n"
        
        # 構建 Few-Shot Prompt
        prompt = self.scale_definition_new + "\n\n【範例句子】（從 dataset_scorer.jsonl 抽取）:\n"
        
        # 按等級分類顯示範例，確保多樣性
        for level in sorted(df_few_shot['score'].unique()):
            level_samples = df_few_shot[df_few_shot['score'] == level]
            level_labels = {1: "溫和", 2: "中性", 3: "不滿", 4: "酸言"}
            level_name = level_labels.get(level, f"Level {level}")
            
            prompt += f"\n【Level {level}：{level_name}】\n"
            # 最多顯示 5 個範例
            for idx, (_, row) in enumerate(level_samples.head(5).iterrows()):
                prompt += f"  • 「{row.get('text', '')}」\n"
        
        return prompt

    def score(self, sentence: str) -> int:
        """呼叫 LLM，對一句話打分數，包含自動 Retry 機制 (對應原本的 score_sentence)"""
        prompt = self.few_shot_prompt + f"\n請對下面句子打分：\n「{sentence}」 →"

        max_retries = 5  
        base_delay = 2   

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                raw_score_content = response.choices[0].message.content
                if raw_score_content is None:
                    return -2
                    
                raw_score = raw_score_content.strip()
                try:
                    score_val = int(raw_score)
                    return score_val
                except:
                    numbers = re.findall(r"\d+", raw_score)
                    score_val = int(numbers[0]) if numbers else -1
                    return score_val
                
            except Exception as e:
                error_str = str(e)
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    wait_time = base_delay * (2 ** attempt) 
                    print(f"\n⚠️ 裁判(Scorer)觸發速率限制，等待 {wait_time} 秒後重試... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n⚠️ 裁判(Scorer) API 呼叫失敗。錯誤: {e}")
                    return -3

        print(f"\n❌ 裁判(Scorer) 重試次數耗盡，放棄此句。")
        return -3