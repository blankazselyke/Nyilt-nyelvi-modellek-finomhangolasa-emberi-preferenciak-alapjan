!pip install datasets trl peft bitsandbytes accelerate transformers wandb

import os
import gc
import torch
import transformers
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import DPOTrainer, DPOConfig setup_chat_format


base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=os.environ['HF_HOME'])
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir=os.environ['HF_HOME'],
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)
model = get_peft_model(model, peft_config)

ref_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir=os.environ['HF_HOME'],
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


df = pd.read_csv('translated_dataset.csv', sep=';', encoding='utf-8-sig')

dpo_dataset_dict = {
    "system": df['system'].tolist(),
    "input": df['input'].tolist(),
    "chosen": df['chosen'].tolist(),
    "rejected": df['rejected'].tolist()
}


dataset = Dataset.from_dict(dpo_dataset_dict)

dataset

!wget -L https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/llama-3-instruct.jinja

chat_template = open('llama-3-instruct.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

def dataset_format(example):
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""
    message = {"role": "user", "content": example['input']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    chosen = example['chosen'] + "<|eot_id|>\n"
    rejected = example['rejected'] + "<|eot_id|>\n"
    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

original_columns = dataset.column_names
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dataset = dataset.map(
    dataset_format,
    remove_columns=original_columns,
    num_proc= os.cpu_count(),
)

model.train()

for param in model.parameters():
    param.requires_grad = True

import wandb
wandb.login()

from sklearn.model_selection import train_test_split
df = dataset.to_pandas()

train_df, test_df = train_test_split(df, test_size=0.2)

train_dataset = dataset.from_pandas(train_df)
test_dataset = dataset.from_pandas(test_df)

train_dataset = train_dataset.remove_columns(["__index_level_0__"])
test_dataset = test_dataset.remove_columns(["__index_level_0__"])

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0 
)

MAX_STEPS = int(3 * len(train_dataset) / 16)

args = DPOConfig(
    output_dir="/project/c_dpo1/magyar_dpo",
    overwrite_output_dir=True,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-6,
    warmup_steps=1000,
    warmup_ratio=0.3,
    do_train=True,
    do_eval=True,
    logging_steps=50,
    eval_steps=50,
    save_total_limit=1,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=MAX_STEPS,
    weight_decay=0.05,
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    max_prompt_length=512,
    beta=0.2,
    report_to="wandb",
    lr_scheduler_type="cosine"
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=512,
    callbacks=[early_stopping_callback]
)

dpo_trainer.train()

final_eval_results = dpo_trainer.evaluate()

dpo_trainer.model.save_pretrained("/project/c_dpo1/magyar_dpo_checkpoint")
tokenizer.save_pretrained("/project/c_dpo1/magyar_dpo_checkpoint")
