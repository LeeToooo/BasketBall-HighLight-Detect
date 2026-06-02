# 🏀 Basketball Highlight Detection

## 📖 Giới thiệu (Introduction)
Đồ án này là một hệ thống tự động nhận diện và trích xuất các khoảnh khắc ghi điểm (Highlights) từ các video bóng rổ. Hệ thống kết hợp nhiều mô hình học sâu để phân tích video và nhận diện các pha bóng gay cấn dựa trên hình ảnh và âm thanh.

## ⚙️ Cách thức hoạt động (How it works)
Hệ thống xử lý video đầu vào thông qua một **Pipeline 3 bước**:

1. **Phát hiện rổ (Rim Detection)**: Sử dụng mô hình nhận diện vật thể **YOLOv8** (`yolov8n.pt`) để quét qua các khung hình (frame) của video nhằm phát hiện vị trí của rổ bóng rổ.
2. **Đánh giá khả năng ghi điểm (Score Classifier)**: Cắt các khung hình chứa rổ bóng rổ (crops) lấy từ bước 1 và đưa vào mô hình phân loại hình ảnh (Image Classifier) - như **ResNet** hoặc **MobileNet** - để dự đoán xác suất khung hình đó có phải là một pha bóng ghi điểm hay không (Point vs No Point).
3. **Trích xuất Highlight (Highlight Extraction)**: Tổng hợp xác suất ghi điểm từ bước 2 cùng với cường độ âm thanh của video (Audio Weight) để xác định các "đỉnh" (peaks) cao trào. Hệ thống sẽ tự động cắt các đoạn clip ngắn xung quanh những điểm cao trào này và ghép lại thành một video highlight hoàn chỉnh.

## 🗂 Cấu trúc thư mục (Project Structure)
- `app.py`: Giao diện Web (UI) sử dụng **Streamlit** để người dùng dễ dàng tải video lên và sử dụng hệ thống.
- `pipeline.py` & `make_highlights.py`: Chứa logic xử lý chính của quy trình, bao gồm chạy inference và thuật toán cắt, ghép video.
- `train_detector.py`: Script dùng để huấn luyện mô hình YOLOv8 cho tác vụ nhận diện rổ bóng rổ.
- `train_classifier.py`: Script dùng để huấn luyện mô hình Image Classifier (phân loại pha có điểm và không có điểm).
- `config.yaml`: File cấu hình các tham số huấn luyện (như learning rate, epochs, cấu trúc model...).
- `requirements.txt`: Chứa danh sách các thư viện Python cần cài đặt.
- `checkpoints/`: Thư mục lưu trữ các trọng số (weights) của các mô hình đã được huấn luyện.
- `data/`: Thư mục chứa dữ liệu hình ảnh, nhãn dùng cho quá trình huấn luyện.
- `outputs/`: Thư mục lưu trữ video highlight và các file CSV kết quả trung gian.

## 🚀 Hướng dẫn cài đặt (Installation)
1. Clone hoặc tải mã nguồn dự án về máy.
2. Cài đặt các thư viện yêu cầu:
```bash
pip install -r requirements.txt
```

## 📥 Tải dữ liệu từ Kaggle (Download Dataset)
Dữ liệu chuẩn của đồ án được cung cấp tại: [Basketball Highlight Detect Dataset](https://www.kaggle.com/datasets/nonamenood/dataset-basketball-highlight-detect/data).

Vui lòng tải dữ liệu thủ công theo các bước sau:
1. Truy cập vào link Kaggle ở trên.
2. Bấm vào nút **Download** (Tải xuống) góc trên bên phải màn hình.
3. Giải nén file tải về và đặt toàn bộ nội dung vào thư mục `data/` trong thư mục gốc của dự án.
(Sau khi giải nén, cấu trúc thư mục của bạn sẽ có dạng `data/crops/`, `data/yolo_rim/`,...)
## 🧠 Hướng dẫn Huấn luyện (Training)

### 1. Huấn luyện mô hình YOLO (Phát hiện rổ)
Chạy lệnh sau để huấn luyện mô hình YOLO nhận diện rổ. Cấu hình dataset được đọc từ `data/yolo_rim/dataset.yaml`.
```bash
python train_detector.py
```
Trọng số (Weights) tốt nhất sau khi huấn luyện sẽ được lưu tại `checkpoints/detector/`.

### 2. Huấn luyện mô hình Phân loại (Classifier)
Bạn có thể tùy chỉnh các tham số huấn luyện như batch_size, learning rate, và loại mô hình (Resnet/Mobilenet) tại file `config.yaml`.
Sau đó chạy lệnh:
```bash
python train_classifier.py
```
Mô hình sẽ sử dụng các ảnh cắt từ rổ trong thư mục `data/crops` để học phân loại. Trọng số sẽ được lưu tại `checkpoints/classifier/`.

## 🎮 Cách sử dụng Ứng dụng (Usage)
Dự án có đi kèm một giao diện Web đơn giản và trực quan.
1. Khởi chạy giao diện bằng lệnh:
```bash
streamlit run app.py
```
2. Trình duyệt sẽ mở ra trang web ứng dụng (Mặc định ở `http://localhost:8501`).
3. **Cách dùng**:
   - Tải lên một đoạn video bóng rổ (`.mp4`, `.avi`, `.mov`).
   - Tinh chỉnh các thông số (Threshold nhận diện rổ, độ tin cậy ghi điểm, ảnh hưởng của âm thanh) ở thanh **Cài đặt Pipeline (MVP)** bên trái nếu muốn.
   - Nhấn **🚀 Chạy Pipeline** và chờ hệ thống xử lý các bước.
   - Khi hoàn thành, ứng dụng sẽ hiển thị video highlight và cung cấp nút **⬇️ Tải xuống**.
