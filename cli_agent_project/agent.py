# agent.py

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help="Enter natural-language CLI instruction.")
    args = parser.parse_args()

    prompt = args.query
    print(f"\nUser Prompt: {prompt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)

    adapter_path = "lora_output"
    if os.path.exists(adapter_path):
        print("✅ Loading LoRA adapter from:", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    else:
        print("⚠️ LoRA adapter not found. Running with base model.")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Step-by-step Plan ===")
    print(generated_text)

    first_line = generated_text.strip().split('\n')[0]
    if first_line.startswith("$") or any(first_line.strip().startswith(cmd) for cmd in ['git', 'tar', 'grep', 'ls', 'python', 'pip']):
        dry_cmd = f"echo {first_line.lstrip('$').strip()}"
        print(f"\n=== Dry-run Command ===\n{dry_cmd}")
        os.system(dry_cmd)

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/trace.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({"query": prompt, "response": generated_text}) + "\n")

    print(f"\n✅ Logged to {log_path}")


if __name__ == "__main__":
    main()
