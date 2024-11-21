from google.colab import drive
drive.mount('/content/drive')

!pip3 install transformers
!pip3 install datasets
!pip3 install trl
!pip install -q -U bitsandbytes
!pip install accelerate
!pip install peft

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, AutoModelForQuestionAnswering
from trl import DPOTrainer, SFTTrainer
from transformers import BitsAndBytesConfig
from torch import bfloat16
import torch
from peft import LoraConfig,PeftModel, PeftConfig, get_peft_model, AutoPeftModelForCausalLM
import warnings
import os
warnings.filterwarnings("ignore")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_name ="facebook/opt-350m"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.config.use_cache = False

def formatting_prompts_func(example):
  output_texts = []
  assert len(example["text"]) == len(example["label"])
  for i in range(len(example['text'])):
    text = f"### Input: ```{example['text'][i]}```\n ### Output: {example['label'][i]}"
    output_texts.append(text)
  return output_texts

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

sft_dataset = load_dataset("gbharti/finance-alpaca", split="train")
sft_datset_new = sft_dataset.select(range(10000))
sft_splitted_data = sft_datset_new.train_test_split(test_size=0.3,seed=42, shuffle=True)

sft_splitted_data["train"] = sft_splitted_data["train"].remove_columns(["input", "text"])
sft_splitted_data["test"] = sft_splitted_data["test"].remove_columns(["input", "text"])

sft_splitted_data["train"] = sft_splitted_data["train"].rename_column("output", "label")
sft_splitted_data["train"] = sft_splitted_data["train"].rename_column("instruction", "text")

sft_splitted_data["test"]  = sft_splitted_data["test"].rename_column("output", "label")
sft_splitted_data["test"]  = sft_splitted_data["test"].rename_column("instruction", "text")

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_path = "/content/drive/MyDrive/fine_tuning_llm_with_dpo_peft/models/"
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["k_proj","v_proj","q_proj","out_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
MAX_STEPS =int(2 * len(sft_splitted_data["train"]) / 4)
args = TrainingArguments(
        output_path,
        overwrite_output_dir=True,
        fp16=True,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        learning_rate=5e-5,
        warmup_steps=500,
        warmup_ratio=0.1,
        do_train=True,
        do_eval=True,
        logging_steps=500,
        eval_steps=500,
        save_total_limit=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=MAX_STEPS,
        weight_decay=0.01,
        dataloader_num_workers=4,
        load_best_model_at_end=True
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=sft_splitted_data["train"],
    eval_dataset=sft_splitted_data["test"],
    tokenizer=tokenizer,
    peft_config=config,
    formatting_func=formatting_prompts_func,
    dataset_text_field="text",
    args=args,         # Trainer arguments
    max_seq_length=256,
)
trainer.train()

output_path = "/content/drive/MyDrive/fine_tuning_llm_with_dpp_peft/models/"
adapter_path = os.path.join(output_path,"checkpoint-3500")

math_dataset = load_dataset("argilla/distilabel-math-preference-dpo", split="train")
def process_dataset(sample_data):
  return {
      "prompt": [f"Question: " + question + "\n\nAnswer: "
      for question in sample_data["instruction"]
      ],
      "chosen": sample_data["chosen_response"],
      "rejected": sample_data["rejected_response"]
  }

original_cols = math_dataset.column_names
math_dataset = math_dataset.map(process_dataset,batched=True,remove_columns=original_cols)

math_dataset_splits = math_dataset.train_test_split(test_size=0.2,seed=42,shuffle=True)
math_dataset_splits

model = AutoPeftModelForCausalLM.from_pretrained(
 adapter_path,
 low_cpu_mem_usage=True,
 torch_dtype=torch.bfloat16,
 load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path,trust_remote_code=True)

MAX_STEPS =int(5 * len(math_dataset_splits["train"]) / 16)

output_path_dpo = os.path.join(output_path, "dpo_models")
args = TrainingArguments(
        output_path_dpo,
        overwrite_output_dir=True,
        fp16=True,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        learning_rate=5e-5,
        warmup_steps=500,
        warmup_ratio=0.1,
        do_train=True,
        do_eval=True,
        logging_steps=500,
        eval_steps=500,
        save_total_limit=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=MAX_STEPS,
        weight_decay=0.01,
        dataloader_num_workers=4,
        load_best_model_at_end=True
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["k_proj","v_proj","q_proj","out_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model,
    args=args,
    beta=0.1,
    train_dataset=math_dataset_splits["train"],
    eval_dataset=math_dataset_splits["test"],
    tokenizer=tokenizer,
    peft_config=config,
    max_length=512,
)
dpo_trainer.train()

!zip -r /content/lib.zip /content/drive/MyDrive/fine_tuning_llm_with_dop_peft/models/checkpoint-3500
!rm -rf /content/drive/MyDrive/fine_tuning_llm_with_dop_peft/models/dpo_models/checkpoint-11500

adapter_model = "/content/drive/MyDrive/fine_tuning_llm_with_dpo_peft/models/dpo_models/checkpoint-500/"
model = AutoPeftModelForCausalLM.from_pretrained(adapter_model, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(adapter_model)

model.eval()
for i in range(len(math_dataset_splits["test"][:20])):
  prompt = math_dataset_splits["test"][i]["prompt"]

  inputs = tokenizer(prompt, return_tensors="pt")
  with torch.no_grad():

    generate_kwargs = dict(
    input_ids=inputs["input_ids"],
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=100,
    repetition_penalty=1.3
    )
    outputs = model.generate(**generate_kwargs)
  print(tokenizer.decode(outputs[0]))
  print("**"*50)

general_prompts = ["Hello! How are you today?", "Solve this mathematical problem: what is 5 + 2? For example, the answer to what is 5 + 1 is: 5 + 1 = 6. Think logically and step by step.", "What should I invest my money in?", "Please tell me your favourite joke.", "Why is it impossible to divide by 0?", "Please give me a recipe for lunch that contains tomato."]

for i in range(len(general_prompts)):
  prompt = general_prompts[i]

  inputs = tokenizer(prompt, return_tensors="pt")
  with torch.no_grad():

    generate_kwargs = dict(
    input_ids=inputs["input_ids"],
    temperature=0.35,
    top_p=0.45,
    top_k=20,
    max_new_tokens=50,
    repetition_penalty=1.5
    )
    outputs = model.generate(**generate_kwargs)
  print(tokenizer.decode(outputs[0]))
  print("**"*50)