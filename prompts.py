# prompts.py
import json
import random
import os

#
class BasicPromptBuilder1:
    """
    只使用態度與邏輯和 Few-Shot 
    """
    def __init__(self):
        self.level_defs = {
            1: "【Level 1：溫和】(重點在對方感受，語氣柔和)",
            2: "【Level 2：中性】(陳述事實，冷靜直接)",
            3: "【Level 3：不滿】(明顯不耐煩，帶情緒壓力)",
            4: "【Level 4：酸】(反語、假稱讚、陰陽怪氣)"
        }

    def build_prompt(self, original_text: str, target_level: int, original_level: int = None, max_chars: int = 60) -> str:
        """
        回傳語氣改寫 Prompt
        
        參數：
            original_text: 原始句子
            target_level: 要改寫為的目標等級 (1-4)
            original_level: 原句的原始等級 (1-4)，如果提供則說明是等級轉換
            max_chars: 最大字數限制
        """
        target_desc = self.level_defs.get(target_level, self.level_defs[2])
        examples_str = ""  # BasicPromptBuilder1 不使用 few-shot 範例

        if original_level is not None and original_level != target_level:
            # 等級轉換模式
            level_labels = {1: "一級", 2: "二級", 3: "三級", 4: "四級"}
            original_level_name = level_labels.get(original_level, f"第{original_level}級")
            target_level_name = level_labels.get(target_level, f"第{target_level}級")
            
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
原句已被評定為【第 {original_level} 級：{original_level_name}】語氣。
請將原句改寫為【第 {target_level} 級：{target_level_name}】的語氣。

### 目標語氣與態度：
{target_desc}

### 風格參考 (請學習其邏輯)：
{examples_str}

### 嚴格執行規則：
1. **保留原意**：改寫時必須保留原句的核心意思，只改變語氣層級。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。
4. **長度限制**：{max_chars} 字以內。

### 開始改寫：
原始句子（第 {original_level} 級）："{original_text}"
改寫為第 {target_level} 級：
請只輸出改寫後句子，勿加說明。
"""
        else:
            # 普通改寫模式
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
請將使用者的「原始句子」改寫為符合「Level {target_level}」的語氣。

### 目標語氣與態度：
{target_desc}

### 風格參考 (請學習其邏輯)：
{examples_str}

### 嚴格執行規則：
1. **保留原意**：**絕對不要回答問題**。如果原句是問句，改寫後也要是問句（或反問）。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。
4. **長度限制**：{max_chars} 字以內。

### 開始改寫：
原始句子："{original_text}"
Level {target_level} 改寫："""



class DataPromptBuilder1:
    """
    自訂 4 句 Few-shot 範例的 Prompt Builder
    """
    # 這裡保留 json_path 和 sample_size 參數，是為了讓你 main.py 不用改程式碼也不會報錯
    def __init__(self, json_path=None, sample_size=None):
        self.level_defs = {
            1: "【Level 1：溫和】(重點在對方感受，語氣柔和)",
            2: "【Level 2：中性】(陳述事實，冷靜直接)",
            3: "【Level 3：不滿】(明顯不耐煩，帶情緒壓力)",
            4: "【Level 4：酸】(反語、假稱讚、陰陽怪氣)"
        }
        self.levels = [1, 2, 3, 4]
        
        # 🔻 在這裡填入想要指定的每一級 x 句 few-shot 範例 🔻
        self.custom_examples = {
            1: [
                "請多指教，很高興認識你。",
                #"希望能買到你喜歡的東西。",
                #"別給自己太大壓力，放輕鬆。",
            ],
            2: [
                "我晚一點才會回到家。",
                #"這兩件衣服的款式差不多。",
                #"我對海鮮類的食物稍微過敏。",
            ],
            3: [
                "這種爛攤子你自己想辦法收拾。",
                #"我真的受夠了你這種漫不經心的態度。",
                #"這種低級的錯誤你也犯，真的太誇張了。",

            ],
            4: [
                "對不起 我不跟一坨跳動的肉說話",
                #"大腦跟大腸雖然很像，但也不能都拿來裝屎阿",
                #"我小時候被狗咬過，所以現在看你有點害怕。",


            ]
        }

    def build_prompt(self, original_text: str, target_level: int, original_level: int = None) -> str:
        """
        使用固定的自訂範例來構建 Prompt
        """
        if target_level not in self.levels:
            return "Error: Level must be 1, 2, 3, or 4."

        target_desc = self.level_defs.get(target_level)
        level_labels = {1: "一級", 2: "二級", 3: "三級", 4: "四級"}
        target_level_name = level_labels.get(target_level, f"第{target_level}級")
        
        # 抓取對應等級的 4 筆指定範例
        selected_examples = self.custom_examples.get(target_level, [])
        
        examples_str = ""
        if selected_examples:
            examples_str = f"以下是此語氣的 {len(selected_examples)} 筆【參考範例句】(請學習其用詞與風格)：\n"
            for ex in selected_examples:
                examples_str += f"- {ex}\n"
        else:
            examples_str = "(此等級目前無參考資料，請依定義發揮)\n"

        if original_level is not None and original_level != target_level:
            # 等級轉換模式
            original_level_name = level_labels.get(original_level, f"第{original_level}級")
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
原句已被評定為【第 {original_level} 級：{original_level_name}】語氣。
請將原句改寫為【第 {target_level} 級：{target_level_name}】的語氣。

### 目標語氣：
{target_desc}

### 參考範例（{target_level_name}風格）：
{examples_str}
### 嚴格執行規則：
1. **保留原意**：改寫時必須保留原句的核心意思，只改變語氣層級。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。

### 開始改寫：
原始句子（第 {original_level} 級）："{original_text}"
改寫為第 {target_level} 級："""
        else:
            # 普通改寫模式
            level_label = level_labels.get(target_level, f"第{target_level}級改寫")
            return f"""你是一位語氣改寫專家。
請將「原句」改寫為「{level_label}」版本。

### 目標語氣：
{target_desc}

### 參考範例：
{examples_str}
### 你的任務：
原句："{original_text}"
{level_label}：
請只輸出改寫後句子，勿加說明。
"""