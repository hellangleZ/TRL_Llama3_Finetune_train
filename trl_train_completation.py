#导入库
import torch
from accelerate import PartialState
from trl import SFTTrainer
from datasets import  load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
#from random import randrange

#在线量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#并行
device_string = PartialState().process_index
device_map={'':device_string}

#模型load
tokenizer = AutoTokenizer.from_pretrained("/aml/new") #Your model local dir, or huggingface repo-id
model = AutoModelForCausalLM.from_pretrained("/aml/new",
                                             quantization_config=bnb_config,
                                             device_map=device_map,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2"
                                            )

#tp != 1不等于1的值将激活更准确但更慢的线性层计算，更好地匹配原始概率，前提是了解之前的pretain）TP配置，不懂就写1
model.config.pretraining_tp = 2


#lora配置
peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
)

#模型应用配置
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

#超参
train_args = TrainingArguments(
    output_dir="/aml/trl",#Your model output directory
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    optim='adamw_bnb_8bit',
    logging_steps=1,
    save_strategy='epoch',
    learning_rate=float('2e-5'),
    bf16=True,    # use bfloat16 for precision training  A10 support
    tf32=True,    # use tf32 for precision training A10 support
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type='constant',
)

# Load dataset
dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train") #Your dataset name of Huggingface

#print(f"dataset size: {len(dataset)}")
#print(dataset[randrange(len(dataset))])


#定义数据格式
def format_instruction(sample):
	return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['instruction_zh']}

### Response:
{sample['output_zh']}
"""

#训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048, #Seq_num
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=train_args,
)

#开始训练
trainer.train()




