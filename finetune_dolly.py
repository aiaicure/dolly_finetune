# Create Instruct Pipeline
import logging
import re

import numpy as np
from transformers import Pipeline, PreTrainedTokenizer
import torch
import dill as dill
import sys

torch.cuda.empty_cache()

logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self, *args, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs
    ):
        super().__init__(*args, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)

    def _sanitize_parameters(self, return_instruction_text=False, **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
            "return_instruction_text": return_instruction_text,
        }

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )[0].cpu()
        instruction_text = model_inputs.pop("instruction_text")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "instruction_text": instruction_text}

    def postprocess(self, model_outputs, response_key_token_id, end_key_token_id, return_instruction_text):
        sequence = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]

        # The response will be set to this variable if we can identify it.
        decoded = None

        # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
        if response_key_token_id and end_key_token_id:
            # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
            # prompt, we should definitely find it.  We will return the tokens found after this token.
            response_pos = None
            response_positions = np.where(sequence == response_key_token_id)[0]
            if len(response_positions) == 0:
                logger.warn(f"Could not find response key {response_key_token_id} in: {sequence}")
            else:
                response_pos = response_positions[0]

            if response_pos:
                # Next find where "### End" is located.  The model has been trained to end its responses with this
                # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                # this token, as the response could be truncated.  If we don't find it then just return everything
                # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                end_pos = None
                end_positions = np.where(sequence == end_key_token_id)[0]
                if len(end_positions) > 0:
                    end_pos = end_positions[0]

                decoded = self.tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()
        else:
            # Otherwise we'll decode everything and use a regex to find the response and end.

            fully_decoded = self.tokenizer.decode(sequence)

            # The response appears after "### Response:".  The model has been trained to append "### End" at the
            # end.
            m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

            if m:
                decoded = m.group(1).strip()
            else:
                # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                # return everything after "### Response:".
                m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                if m:
                    decoded = m.group(1).strip()
                else:
                    logger.warn(f"Failed to find response in:\n{fully_decoded}")

        if return_instruction_text:
            return {"instruction_text": instruction_text, "generated_text": decoded}

        return decoded

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")

model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b",
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)

step1 = list(model.state_dict().keys())
#sys.exit()


from datasets import load_dataset

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


### Finetuning dolly
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


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
step2 = list(model.state_dict().keys())

#print(f"Are they same? {sorted(step1) == sorted(step2)}")
#sys.exit()
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
step3 = list(model.state_dict().keys())
#print(f"Are they same? {sorted(step2) == sorted(step3)}")
#print(sorted(step3))
#sys.exit()
#print(f"Print Model Config: {model.config}")
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
#trainer.model._save_pretrained(save_directory="alpaca-lora-dolly-2.0")

#model.save_pretrained("alpaca-lora-dolly-2.0")
#tokenizer.save_pretrained('alpaca-lora-dolly-2.0')
#model.save('alpaca-lora-dolly-2.0')
#torch.save(model, 'alpaca-lora-dolly-2.0', pickle_module=dill)
model.config.to_json_file("config.json")
torch.save(model.state_dict(), 'alpaca-lora-dolly-2.0.pth')
print(len(model.state_dict().keys()))
#torch.save(model.state_dict(), 'alpaca-lora-dolly-2.0.pth')
#generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

#data_point = {"instruction": "Summarize the following note as if you were an experienced medical doctor", "input": "T/Sicu Nsg Progress Note\n0700>>1900\n\nEvents- US and tap of left hip>>>no significant fluid obtained\n        slow wean of levophed underway>>>tolerating with MAP's>65\n        TPN to start tonight\n\nNeuro- perrl; responds to pain with facial grimacing; no movemet of extremities spntaneously or to pain. Is opening eyes slightly more to stimualtion/voice, but no attempts to communicate or respond.\nFentanyl 50mcq prn for discomfort.\n\ncvs- see careview for specifics..generally no labile swings. Levophed being weaned slowly- @ .15mcq/kg/min(.22) with maps>65\n***fluid bolus this am(500ccx2) well tolerated w/o increase in fp's\n\nresp- no vent changes today; tolerating PSV 20, PEEP 17, 50% fio2\n      RR 20-30 with tidal volumes 500-600cc\n      Breath sounds are clear/diminished\n        abg's at baseline range.\n        scant secretions.. thick white\n\nrenal- u/o ~ 50-60/hr amber w/sediment\n       cont in pos fluid balance\n       electrolytes repleted as needed\n     **bun/creat cont to elevate\n\nid- afebrile today with wbc >20.. multiple antibiotics cont. Vanco x1 dose today and fluconazole addedd to regimen.\n\nheme- no issues\n\ngi- obese distended abd/soft. Hypoactive bowel sounds on occ\n    rectal tube in place with thick soft brown stool...irrigation prn to clear stool.\n    npo with gastric tube  replaced orally with improved drainage of copious bilious fluid.\n    protonix.\n  **TPN to satert tonight\n\nendo- insulin drip 4-6u/hr to maintian blood sugars <150\n\nskin- no change in right middle finger or right foot wound..see careview assessment.\n      sc heparin increased to tid dosing\n      compression boot to LLE in use\n   *** dopplerable pt/dp pulses. Feet are cold with mottled toe/heel areas and with decreased cap refill\n      increasing edema in perineal area..no breakdown observed.\n\nassess- no major changes/events today\n        maintaining multi-system supports/weaning as tolerated\n\nplan- cont with current mngmnt\n      comfort mngmnt for pain/anxiety prn\n      family support and condition updates.\n", "output": "The patient is in the T/Sicu and has undergone a tap of the left hip with no significant fluid obtained. Levophed is being slowly weaned and the patient is tolerating it with MAP's>65. TPN will start tonight. Neurologically, the patient is perrl and responds to pain with facial grimacing, but there is no movement of extremities spontaneously or to pain. The patient is opening their eyes slightly more to stimulation/voice, but there are no attempts to communicate or respond. Fentanyl is being administered for discomfort. CVS is stable with no labile swings. Respiratory-wise, there are no vent changes and the patient is tolerating PSV 20, PEEP 17, and 50% fio2. Renally, the patient has a u/o of ~50-60/hr amber w/sediment and is in a positive fluid balance. Electrolytes are being repleted as needed, but the bun/creat continues to elevate. The patient is afebrile today with a wbc >20 and multiple antibiotics are being continued. Heme is stable. GI-wise, the patient has an obese distended abd/soft and hypoactive bowel sounds on occasion. The patient is npo with a gastric tube replaced orally with improved drainage of copious bilious fluid. Endocrinologically, the patient is on an insulin drip to maintain blood sugars <150. The skin is stable with no change in the right middle finger or right foot wound. The patient has cold feet with mottled toe/heel areas and decreased cap refill. There is increasing edema in the perineal area with no breakdown observed. The patient is maintaining multi-system supports/weaning as tolerated. The plan is to continue with current management, provide comfort management for pain/anxiety prn, and provide family support and condition updates."}
#generate_text(data_point)
#print(generate_text(data_point['instruction'] + ':' + data_point['input']))
