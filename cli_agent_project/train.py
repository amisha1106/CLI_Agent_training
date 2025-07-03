# train.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Load dataset
data = load_dataset("json", data_files="data/cli_qa_pairs.json")
def tokenize_fn(example):
    return tokenizer(example['question'] + "\n" + example['answer'], truncation=True, padding="max_length", max_length=256)

tokenized = data["train"].map(tokenize_fn)

# Training arguments
args = TrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
)

dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=dc)
trainer.train()
model.save_pretrained("lora_output")
print("✅ Training complete")