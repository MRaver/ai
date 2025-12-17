import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import sys

# --- CẤU HÌNH ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Model gốc
ADAPTER_PATH = "./lora_model"                  # Thư mục chứa LoRA bạn vừa giải nén

print("⏳ Đang khởi động AI Pentest Assistant...")
print("1. Đang tải model gốc (Qwen 2.5 0.5B)...")

# 1. Load Model Gốc
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16, # Dùng float16 cho nhẹ
        device_map="auto"          # Tự chọn GPU hoặc CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
except Exception as e:
    print(f"❌ Lỗi load model gốc: {e}")
    sys.exit(1)

print("2. Đang gắn não LoRA (Fine-tuned)...")

# 2. Load LoRA Adapter
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval() # Chuyển sang chế độ dùng thử (Inference)
except Exception as e:
    print(f"❌ Lỗi không tìm thấy thư mục '{ADAPTER_PATH}'. Bạn đã giải nén file zip chưa?")
    sys.exit(1)

print("✅ KHỞI ĐỘNG THÀNH CÔNG! Sẵn sàng quét lỗ hổng.")
print("="*60)

# --- HÀM XỬ LÝ ---
def scan_vulnerability(code_snippet):
    # System Prompt y hệt lúc train
    system_prompt = """You are a cybersecurity expert. 
Analyze the input and identify the web vulnerability.
Return the answer strictly in JSON format with these keys:
- vulnerability
- explanation
- example_payload
- mitigation
Do not output any text outside the JSON block."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: Analyze the following input and identify any web vulnerability.\n\nInput to analyze:\n{code_snippet}"}
    ]

    # Format input theo chuẩn ChatML
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # Nhiệt độ thấp để output ổn định
            do_sample=True,
            top_p=0.9
        )

    # Cắt bỏ phần prompt, chỉ lấy câu trả lời mới sinh ra
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# --- VÒNG LẶP CHÍNH ---
while True:
    print("\n🔹 Nhập đoạn code/URL cần check (Gõ 'exit' để thoát):")
    print("(Nhấn Enter 2 lần để gửi)")
    
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "": break
        if line.lower() == "exit": sys.exit()
        lines.append(line)
    
    user_input = "\n".join(lines)
    
    if not user_input.strip(): continue

    print("\n🔍 Đang phân tích...")
    result = scan_vulnerability(user_input)
    
    print("-" * 20 + " KẾT QUẢ " + "-" * 20)
    print(result)
    
    # Thử parse JSON để kiểm tra độ xịn
    try:
        parsed = json.loads(result)
        print("\n✅ [JSON Valid] Phát hiện lỗ hổng:", parsed.get('vulnerability', 'Unknown'))
    except:
        print("\n⚠️ Output chưa chuẩn JSON lắm (nhưng vẫn đọc được).")