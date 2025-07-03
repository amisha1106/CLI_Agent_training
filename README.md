# CLI_Agent_training


CLI Agent - End-to-End Workflow

Overview

This project fine-tunes a lightweight language model on CLI-related Q&A data using LoRA, wraps it in an agent that interprets natural language commands, and supports dry-run shell execution.

Setup Instructions (Google Colab)

1. Clone or Upload Project
Upload cli_agent_project.zip to Colab and unzip:

!unzip cli_agent_project.zip
%cd cli_agent_project

2. Install Dependencies
!pip install -q transformers datasets peft accelerate

3. Train Model
!python train.py

4. Run CLI Agent
!python agent.py --query "Create a new Git branch and switch to it"

Files

train.py: Fine-tunes TinyLlama with LoRA.
agent.py: Accepts a CLI query, returns plan, logs dry-run.
data/cli_qa_pairs.json: 150+ Q&A examples.
lora_output/: Stores adapter after training.
logs/trace.jsonl: Logs of queries and generated plans.
eval_static.md: Base vs fine-tuned result comparison.
eval_dynamic.md: Evaluation on 5+2 test prompts.
report.md: One-page summary report.

Requirements
Google Colab (free T4 GPU)
Python 3.10+
LoRA-compatible model (TinyLlama, <2B)

Credits
TinyLlama 1.1B Model
HuggingFace Transformers
PEFT for LoRA



