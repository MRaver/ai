import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Khuyên dùng float16 để chạy nhanh hơn và tốn ít VRAM hơn
)

# 2. Load dataset
try:
    with open("web_pentest_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print("Không tìm thấy file dataset. Hãy đảm bảo bạn đã lưu file JSON.")
    dataset = []

# Định nghĩa System Prompt để ép model trả về đúng định dạng JSON
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

for i, sample in enumerate(dataset[:3]): # Test thử 3 mẫu đầu
    instruction = sample["instruction"]
    input_text = sample["input"]

    # Cấu trúc messages chuẩn cho Qwen
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {instruction}\n\nInput to analyze:\n{input_text}"}
    ]

    # Sử dụng apply_chat_template để thêm các token đặc biệt (<|im_start|>, etc.)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,      # Đủ dài cho JSON output
        temperature=0.1,         # Để thấp (0.1) giúp model tập trung vào logic và đúng format hơn
        do_sample=True,
        top_p=0.9
    )

    # Cắt bỏ phần input (prompt) khỏi output
    response = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True
)


    print(f"\n{'='*20} Sample {i+1} {'='*20}")
    print(f"Input Code: {input_text[:50]}...") # In gọn input
    print("-" * 10 + " Model Output " + "-" * 10)
    print(response)
    
    # (Optional) Thử parse JSON để xem model trả về có đúng chuẩn không
    try:
        json_resp = json.loads(response)
        print("\n✅ Valid JSON Format detected.")
    except:
        print("\n⚠️  Output is not valid JSON (Common with small models like 0.5B).")