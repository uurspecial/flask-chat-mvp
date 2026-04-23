# 雞掰風格調整

[scorer分類器](https://docs.google.com/presentation/d/1Zpcz8_jk9SJX_LGTDvEewFSUnxJUQxOkiZLtQi42qyo/edit?usp=sharing)

---
# 🎭 Ji-Bai Tuner: LLM Tone-Transfer Benchmark (語氣轉換對照實驗框架)

這是一個專注於**「可控文本語氣轉換 (Controllable Tone Transfer)」**的自動化對照實驗專案。
本專案的目標是讓大型語言模型 (LLMs) 能精準控制回覆的「語氣等級」（從溫和有禮到高級酸民），並透過嚴謹的 `LLM-as-a-Judge` 機制進行自動評分與 **重試 (Retry) 修正** 邏輯。

---

## 📂 專案結構導覽 (Repository Structure)

本專案包含多個開發階段的紀錄，核心運作邏輯集中於 `Generator/` 目錄：

### 🟢 核心運作區 (Active Modules)
* **`Generator/`**：**專案核心。** 包含最新的多模型生成與自動評分框架。
  * `main.py`：實驗主程式，負責跑迴圈、觸發 Retry 機制並輸出 CSV 結果。
  * `generators.py`：統一封裝 `Gemini 2.5 Flash`, `Breeze-7B`, `Llama-3-8B` 的呼叫邏輯。
  * `prompts.py`：動態提示詞工程，支援規則 Prompt 與基於資料庫抽樣的 Data Prompt。
  * `scorer.py`：基於 `GPT-4o-mini` 的自動裁判，負責為產出句子進行 1~4 級評分。
  * `utils.py`：負責環境變數載入與通用工具函數。
* **`dataset/`**：實驗所需的資料集，嚴格區分用途以避免訓練/測試資料重疊。
  * `dataset_scorer.jsonl`：用於 Scorer (裁判) 學習標準的 Few-Shot 資料。
  * `dataset_test.jsonl`：Generator 生成測試專用的句子（與 Scorer 範例完全不重複）。
  * `dataset1.jsonl`：原始完整資料集。
* **`requirements.txt`**：專案執行所需的所有 Python 依賴套件。

### 🟡 開發中 / 原型區 (WIP / Prototypes)
* **`ji-bai-tuner/`**：網頁端原型 (Web UI)。基於 Flask 的互動介面，包含「雞掰度拉桿」。目前重心在後端 Benchmark，未來將與核心模組整合。

### 🟤 歷史紀錄 / 已荒廢 (Deprecated / Historical)
* **`chinese-roberta-wwm-ext/`**：早期嘗試使用小模型進行微調 (Fine-tuning) 的實驗紀錄。
* **`few-shot/`**：早期手動測試 Few-shot 分類的腳本。
* **`sentence/`**：紀錄 Generator 模組從無到有的開發草稿與演進過程。

---

## 🛠️ 快速啟動指南 (Quick Start)

### 1. 安裝環境與套件
建議使用 Python 3.10+ 環境。在終端機執行：

```bash
# 複製專案
git clone [https://github.com/uurspecial/flask-chat-mvp.git](https://github.com/uurspecial/flask-chat-mvp.git)
cd flask-chat-mvp

# 安裝套件
pip install -r requirements.txt

