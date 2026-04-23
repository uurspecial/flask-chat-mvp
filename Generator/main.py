# run_comparison.py
import pandas as pd
import os
import json
import random
import utils

from generators import BreezeGenerator, LlamaGenerator, GroqLlama318B
from scorer import Scorer
from prompts import BasicPromptBuilder1, DataPromptBuilder1
from huggingface_hub import login
login(token="")
from dotenv import load_dotenv
load_dotenv()

# === 參數設定 ===
current_dir = os.path.dirname(os.path.abspath(__file__))

# === 加載資料集 ===
def load_sentences_from_dataset(batch_size=189):
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(parent_dir, "dataset_test_generator.jsonl")
    
    sentences = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sentences.append({
                    "original_text": data["text"],
                    "target_level": int(data["target"])
                })

    if not sentences:
        return []

    batch = min(batch_size, len(sentences))
    return random.sample(sentences, batch)

SENTENCES = load_sentences_from_dataset()

def main():
    api_key = utils.setup_env()
    
    breeze_gen = BreezeGenerator() 
    llama_gen = LlamaGenerator()
    groq_8b_gen = GroqLlama318B()

    scorer = Scorer(api_key, model_name="gpt-4o-mini")
    basic_prompter = BasicPromptBuilder1()
    data_prompter = DataPromptBuilder1()
    
    models = {
        "Groq-8B": groq_8b_gen,
        "Breeze": breeze_gen,
        "Llama": llama_gen
    }
    
    results = []
    print(f"開始執行，共 {len(SENTENCES)} 句...")
    
    for i, sentence_dict in enumerate(SENTENCES, 1):
        original_text = sentence_dict["original_text"]
        target_level = sentence_dict["target_level"]
        
        seq_id = f"S{i:03d}"
        print(f"\n{'='*60}")
        print(f"Processing {seq_id}: {original_text}")
        print(f"  → 目標生成等級: {target_level}級")
        print(f"{'='*60}")
        
        original_score = scorer.score(original_text)
        print(f"  ✓ 原句評分結果: {original_score} 級")
        
        print(f"\n    生成第 {target_level} 級版本...")

        for model_name, generator in models.items():
            print(f"\n    >>> 模型：{model_name}")

            for use_data in [False, True]:
                prompt_type = "Basic" if not use_data else "Data"
                prompter = basic_prompter if not use_data else data_prompter

                prompt = prompter.build_prompt(
                    original_text,
                    target_level=target_level,
                    original_level=original_score
                )

                output_text = generator.generate(custom_prompt=prompt)
                predicted_score = scorer.score(output_text)
                is_correct = (predicted_score == target_level)

                print(f"      [{target_level}級]: {output_text} (Score: {predicted_score}) {'✓' if is_correct else '✗'}")
                results.append({
                    "id": seq_id,
                    "original_text": original_text,
                    "original_score": original_score,
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "use_data": use_data,
                    "target_level": target_level,
                    "generated_text": output_text,
                    "predicted_score": predicted_score,
                    "is_correct": is_correct
                })
        
        print(f"\n  ✓ {seq_id} 完成")

    df = pd.DataFrame(results)
    
    base_filename = "results"
    counter = 1
    current_output_file = os.path.join(current_dir, f"{base_filename}_{counter}.csv")
    
    while os.path.exists(current_output_file):
        counter += 1
        current_output_file = os.path.join(current_dir, f"{base_filename}_{counter}.csv")
    
    df.to_csv(current_output_file, index=False, encoding="utf-8-sig")
    print(f"\n  結果已存至: {current_output_file}")
    
    print(f"\n準確率統計 (模型與模式):")
    df["model_mode"] = df["model"] + "_" + df["prompt_type"]
    stats = df.groupby("model_mode")["is_correct"].mean() * 100
    print(stats)
    
    print("\n 準確率統計 (分等級):")
    print(df.groupby("target_level")["is_correct"].mean() * 100)

if __name__ == "__main__":
    main()