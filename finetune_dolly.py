# Create Instruct Pipeline
from instruct_pipeline import *

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

### Finetuning dolly
from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

torch.cuda.empty_cache()

MODEL_NAME = "databricks/dolly-v2-3b"
SAVED_STATE_DICT = "alpaca-lora-dolly-2.0.pth"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

data = load_dataset("json",
                    data_files="./AlpacaDataCleaned/medical_notes_with_chatgpt_summary.json")

def generate_prompt(data_point):
    # taken from https://github.com/tloen/alpaca-lora
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


data = data.map(lambda data_point: {"prompt": tokenizer(generate_prompt(data_point))})

# Settings for A100 - For 3090
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1 #2  # paper uses 3
LEARNING_RATE = 2e-5
CUTOFF_LEN = 256
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

data = load_dataset("json", data_files="./AlpacaDataCleaned/medical_notes_with_chatgpt_summary.json")

data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir="lora-dolly",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.config.to_json_file("config.json")
torch.save(model.state_dict(), SAVED_STATE_DICT)
