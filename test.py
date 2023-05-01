from peft import LoraConfig, get_peft_model, LoraConfig, get_peft_model
from instruct_pipeline import InstructionTextGenerationPipeline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "databricks/dolly-v2-3b"
SAVED_STATE_DICT = "alpaca-lora-dolly-2.0.pth"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

state_dict = torch.load(SAVED_STATE_DICT)
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
model.load_state_dict(state_dict)
model.eval()

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

data_point = {"instruction": "Summarize the following note as if you were an experienced medical doctor", "input": "T/Sicu Nsg Progress Note\n0700>>1900\n\nEvents- US and tap of left hip>>>no significant fluid obtained\n        slow wean of levophed underway>>>tolerating with MAP's>65\n        TPN to start tonight\n\nNeuro- perrl; responds to pain with facial grimacing; no movemet of extremities spntaneously or to pain. Is opening eyes slightly more to stimualtion/voice, but no attempts to communicate or respond.\nFentanyl 50mcq prn for discomfort.\n\ncvs- see careview for specifics..generally no labile swings. Levophed being weaned slowly- @ .15mcq/kg/min(.22) with maps>65\n***fluid bolus this am(500ccx2) well tolerated w/o increase in fp's\n\nresp- no vent changes today; tolerating PSV 20, PEEP 17, 50% fio2\n      RR 20-30 with tidal volumes 500-600cc\n      Breath sounds are clear/diminished\n        abg's at baseline range.\n        scant secretions.. thick white\n\nrenal- u/o ~ 50-60/hr amber w/sediment\n       cont in pos fluid balance\n       electrolytes repleted as needed\n     **bun/creat cont to elevate\n\nid- afebrile today with wbc >20.. multiple antibiotics cont. Vanco x1 dose today and fluconazole addedd to regimen.\n\nheme- no issues\n\ngi- obese distended abd/soft. Hypoactive bowel sounds on occ\n    rectal tube in place with thick soft brown stool...irrigation prn to clear stool.\n    npo with gastric tube  replaced orally with improved drainage of copious bilious fluid.\n    protonix.\n  **TPN to satert tonight\n\nendo- insulin drip 4-6u/hr to maintian blood sugars <150\n\nskin- no change in right middle finger or right foot wound..see careview assessment.\n      sc heparin increased to tid dosing\n      compression boot to LLE in use\n   *** dopplerable pt/dp pulses. Feet are cold with mottled toe/heel areas and with decreased cap refill\n      increasing edema in perineal area..no breakdown observed.\n\nassess- no major changes/events today\n        maintaining multi-system supports/weaning as tolerated\n\nplan- cont with current mngmnt\n      comfort mngmnt for pain/anxiety prn\n      family support and condition updates.\n", "output": "The patient is in the T/Sicu and has undergone a tap of the left hip with no significant fluid obtained. Levophed is being slowly weaned and the patient is tolerating it with MAP's>65. TPN will start tonight. Neurologically, the patient is perrl and responds to pain with facial grimacing, but there is no movement of extremities spontaneously or to pain. The patient is opening their eyes slightly more to stimulation/voice, but there are no attempts to communicate or respond. Fentanyl is being administered for discomfort. CVS is stable with no labile swings. Respiratory-wise, there are no vent changes and the patient is tolerating PSV 20, PEEP 17, and 50% fio2. Renally, the patient has a u/o of ~50-60/hr amber w/sediment and is in a positive fluid balance. Electrolytes are being repleted as needed, but the bun/creat continues to elevate. The patient is afebrile today with a wbc >20 and multiple antibiotics are being continued. Heme is stable. GI-wise, the patient has an obese distended abd/soft and hypoactive bowel sounds on occasion. The patient is npo with a gastric tube replaced orally with improved drainage of copious bilious fluid. Endocrinologically, the patient is on an insulin drip to maintain blood sugars <150. The skin is stable with no change in the right middle finger or right foot wound. The patient has cold feet with mottled toe/heel areas and decreased cap refill. There is increasing edema in the perineal area with no breakdown observed. The patient is maintaining multi-system supports/weaning as tolerated. The plan is to continue with current management, provide comfort management for pain/anxiety prn, and provide family support and condition updates."}
print(generate_text(data_point['instruction'] + ':' + data_point['input']))
