import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# 2. Load dataset
try:
    with open("data/web_pentest_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print("Không tìm thấy file dataset.")
    dataset = []

# System Prompt
system_prompt = """You are a cybersecurity expert. 
Analyze the input and identify the web vulnerability.
Return the answer strictly in JSON format with these keys:
- vulnerability
- explanation
- example_payload
- mitigation
Do not output any text outside the JSON block."""

# 3. Run inference
print(f"Running inference on {model_name}...\n")

for i, sample in enumerate(dataset[:3]):
    instruction = sample["instruction"]
    input_text = sample["input"]

    # --- BƯỚC 1: CHUẨN BỊ INPUT ---
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {instruction}\n\nInput to analyze:\n{input_text}"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Lấy độ dài input để lát nữa cắt
    input_len = model_inputs.input_ids.shape[1]

    # --- BƯỚC 2: GENERATE (CHỈ 1 LẦN DUY NHẤT) ---
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,  # Giữ nhiệt độ thấp cho code/json
        do_sample=True,
        top_p=0.9
    )

    # --- BƯỚC 3: CẮT TOKEN & DECODE ---
    # Chỉ lấy phần token mới sinh ra (bỏ qua phần input ban đầu)
    generated_ids_trimmed = generated_ids[:, input_len:]
    
    response = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)

    # --- BƯỚC 4: IN KẾT QUẢ ---
    print(f"\n{'='*20} Sample {i+1} {'='*20}")
    print(f"Input Code: {input_text[:50]}...") 
    print("-" * 10 + " Model Output " + "-" * 10)
    print(response)
    
    # Check JSON
    try:
        json_resp = json.loads(response)
        print("\n Valid JSON Format.")
    except:
        print("\n  Invalid JSON.")