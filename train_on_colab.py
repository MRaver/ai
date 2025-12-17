
# Cell 1: Upload Data
from google.colab import files
import os

if not os.path.exists("train_data_for_finetune.json"):
    print("⬆️ Vui lòng chọn file 'train_data_for_finetune.json'...")
    uploaded = files.upload()
    print("✅ Đã upload xong!")
else:
    print("✅ File đã có sẵn, sang Cell 2 train thôi!")
# Cell 2
# 1. Cài đặt thư viện
%%capture
!pip install unsloth
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# --- FIX LỖI TREO (QUAN TRỌNG) ---
import os
# Tắt cái hỏi đăng nhập WandB đi để nó không đợi mãi
os.environ["WANDB_DISABLED"] = "true" 
# ---------------------------------

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from google.colab import files

# 2. Cấu hình Model
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Cấu hình LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Load Data
# Đảm bảo bạn đã upload file này lên cột bên trái nhé
dataset = load_dataset("json", data_files="train_data_for_finetune.json", split="train")

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("🚀 Đang train model... (Lần này sẽ hiện thanh % chạy vèo vèo)")
trainer_stats = trainer.train()

# 6. Lưu và Tải về
print("✅ Train xong! Đang nén file...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

!zip -r lora_model.zip lora_model
files.download('lora_model.zip')