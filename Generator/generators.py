# generators_2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types
import os
import prompts
import time
from google.api_core import exceptions
from google.genai import types
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

class BaseGenerator:
    """
    統一的生成器基底類別，確保所有模型介面一致、取得的 Prompt 絕對相同。
    """
    def __init__(self, model_name):
        self.model_name = model_name
        # 實例化兩個 Prompt Builder
        self.basic_prompter = prompts.BasicPromptBuilder1()
        # 這裡明確指定讀取 "dataset_test.jsonl"
        self.data_prompter = prompts.DataPromptBuilder1("dataset_test.jsonl")

    def _get_unified_prompt(self, text, level, use_data):
        if use_data:
            return self.data_prompter.build_prompt(text, level)
        else:
            return self.basic_prompter.build_prompt(text, level)

    def _clean_output(self, raw_text):
        if not raw_text:
            return ""

        text = raw_text.strip()

        # 若模型回傳含有答案前綴，優先截取具體改寫句
        import re
        m = re.search(r"改寫為第\s*\d+\s*級[:：]\s*(.+)", text)
        if m:
            text = m.group(1).strip()

        # 避免包含冗長說明與解析, 通常最終輸出在最後一行或第一行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        candidate = lines[-1]

        # 如果最後一行看起來像是補述（例如"這個改寫保留了原句"），則使用第一行
        if len(lines) > 1 and any(phrase in candidate for phrase in ["保留", "這個改寫", "原句"]):
            candidate = lines[0]

        # 避免單信英文噪音: 若返還純 emoji 或英文簡短詞，保留原始但這情況少
        # 若候選答案只有 ASCII 文字，嘗試選擇含中文的行
        ascii_pattern = r"^[A-Za-z0-9\s\.,!?;:'\"()\-]+$"
        if re.fullmatch(ascii_pattern, candidate):
            for line in lines:
                if not re.fullmatch(ascii_pattern, line):
                    candidate = line
                    break

        return candidate.strip()

    def generate(self, text=None, level=None, use_data=False, temperature=0.5, max_new_tokens=100, custom_prompt=None):
        # 子類別必須實作此方法
        # custom_prompt: 如果提供，直接使用此 prompt；否則根據 text 和 level 構建
        raise NotImplementedError

# === Breeze 生成器 ===
class BreezeGenerator(BaseGenerator):
    def __init__(self, model_name="MediaTek-Research/Breeze-7B-Instruct-v1_0"):
        super().__init__(model_name)
        print(f"🔄 正在載入 Breeze 模型: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate(self, text=None, level=None, use_data=False, temperature=0.5, max_new_tokens=100, custom_prompt=None):
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._get_unified_prompt(text, level, use_data)
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_ids = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_ids, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0][inputs_ids.input_ids.shape[1]:], skip_special_tokens=True)
        raw = (decoded or "").strip()
        return self._clean_output(raw)


# === Llama 生成器 ===
class LlamaGenerator(BaseGenerator):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"): 
        super().__init__(model_name)
        print(f" 正在載入 Llama 模型: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Llama 家族模型經常沒有預設的 pad_token，這裡設定以防報錯
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate(self, text=None, level=None, use_data=False, temperature=0.5, max_new_tokens=100, custom_prompt=None):
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._get_unified_prompt(text, level, use_data)
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_ids = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_ids, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0][inputs_ids.input_ids.shape[1]:], skip_special_tokens=True)
        raw = (decoded or "").strip()
        return self._clean_output(raw)


class Llama70Generator(BaseGenerator):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
        super().__init__(model_name)
        print(f" 正在載入 Llama 模型: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Llama 家族模型經常沒有預設的 pad_token，這裡設定以防報錯
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
       )

    def generate(self, text=None, level=None, use_data=False, temperature=0.5, max_new_tokens=100, custom_prompt=None):
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._get_unified_prompt(text, level, use_data)
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_ids = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_ids, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0][inputs_ids.input_ids.shape[1]:], skip_special_tokens=True)
        raw = (decoded or "").strip()
        return self._clean_output(raw)
class GroqLlama318B(BaseGenerator):
    """Groq Llama 3.1 8B - 快速、低成本"""
    def __init__(self, api_key=None):
        super().__init__("llama-3.1-8b-instant")
        
        if not HAS_GROQ:
            raise ImportError("❌ groq 套件未安裝。請執行: pip install groq")
        
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("❌ 找不到 GROQ_API_KEY。請設定環境變數或傳入 api_key 參數。")
        
        self.client = Groq(api_key=key)
        print(f"✅ 已初始化 Groq 生成器: {self.model_name}")
    
    def generate(self, text=None, level=None, use_data=False, temperature=0.7, max_new_tokens=100, custom_prompt=None):
        """生成文本"""
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self._get_unified_prompt(text, level, use_data)
        
        try:
            message = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            
            result = message.choices[0].message.content.strip()
            return result
            
        except Exception as e:
            print(f"❌ Groq 生成錯誤 ({self.model_name}): {e}")
            return "Error"