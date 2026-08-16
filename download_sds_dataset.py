import os
import shutil
import kagglehub

def download_and_move_dataset():
    print("🚀 Đang khởi tạo kết nối tải SeaDroneSee dataset từ Kaggle...")
    
    # kagglehub.dataset_download sẽ tải về thư mục cache mặc định (vd: ~/.cache/kagglehub)
    # Hàm này trả về đường dẫn chính xác nơi chứa dữ liệu vừa tải
    cache_path = kagglehub.dataset_download("ubiratanfilho/sds-dataset")
    print(f"\n✅ Đã tải xong! Dữ liệu gốc nằm tại cache: {cache_path}")

    # Đường dẫn thư mục đích trong dự án
    dest_path = "./sds_dataset"

    # Kiểm tra và copy sang thư mục của dự án để dễ quản lý
    if not os.path.exists(dest_path):
        print(f"\n📂 Đang di chuyển dữ liệu về thư mục dự án ({dest_path})...")
        shutil.copytree(cache_path, dest_path)
        print(f"🎉 Hoàn tất! Toàn bộ dataset đã được lưu tại: {os.path.abspath(dest_path)}")
    else:
        print(f"\n⚠️ Thư mục '{dest_path}' đã tồn tại. Bỏ qua bước copy.")
        print(f"Nếu muốn tải lại, hãy xóa thư mục '{dest_path}' trước.")

if __name__ == "__main__":
    download_and_move_dataset()

