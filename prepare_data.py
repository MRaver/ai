import json
import os

# --- CẤU HÌNH ---
# Tên file dataset gốc bạn đã tạo (đảm bảo file này đang nằm đúng chỗ)
input_path = "data/web_pentest_dataset.json" 

# Tên file kết quả sẽ được tạo ra (dùng file này để train)
output_path = "data/train_data_for_finetune.json"

# --- XỬ LÝ ---
def convert_data():
    # 1. Kiểm tra xem file gốc có tồn tại không
    if not os.path.exists(input_path):
        print(f"❌ LỖI: Không tìm thấy file '{input_path}'")
        print("👉 Hãy chắc chắn bạn đã lưu dataset JSON vào thư mục data/")
        return

    # 2. Đọc dữ liệu gốc
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"🔄 Đang xử lý {len(raw_data)} mẫu dữ liệu...")

    # 3. System Prompt (Khuôn mẫu cho model)
    SYSTEM_PROMPT = """You are a cybersecurity expert. 
Analyze the input and identify the web vulnerability.
Return the answer strictly in JSON format with these keys:
- vulnerability
- explanation
- example_payload
- mitigation
Do not output any text outside the JSON block."""

    formatted_data = []

    for sample in raw_data:
        # User: Gộp Instruction + Input code
        user_content = f"Task: {sample['instruction']}\n\nInput to analyze:\n{sample['input']}"
        
        # Assistant: Output mong muốn (convert Dictionary -> JSON String)
        # ensure_ascii=False để giữ tiếng Việt hoặc ký tự đặc biệt nếu có
        assistant_content = json.dumps(sample['output'], ensure_ascii=False)

        # Tạo cấu trúc hội thoại chuẩn
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        # Thêm vào danh sách
        formatted_data.append({"messages": conversation})

    # 4. Lưu file kết quả
    # Tạo thư mục data nếu lỡ tay xóa
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ XONG! Đã tạo file: {output_path}")
    print(f"📊 Tổng cộng: {len(formatted_data)} mẫu hội thoại.")
    print("👉 Giờ bạn có thể dùng file này để fine-tune model.")

if __name__ == "__main__":
    convert_data()