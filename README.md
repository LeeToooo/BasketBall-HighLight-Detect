# 🏀 Basketball Highlight Detection

## 📖 Giới thiệu (Introduction)
Đồ án này là một hệ thống tự động nhận diện và trích xuất các khoảnh khắc ghi điểm (Highlights) từ các video bóng rổ. Hệ thống kết hợp nhiều mô hình học sâu để phân tích video và nhận diện các pha bóng gay cấn dựa trên hình ảnh và âm thanh.

## 🔗 Tài nguyên & Dữ liệu (Resources & Datasets)
Dự án sử dụng và công khai các tài nguyên sau:
- **Dữ liệu ảnh & crops (Raw Data)**: [Kaggle - Basketball Highlight Detect Dataset](https://www.kaggle.com/datasets/nonamenood/dataset-basketball-highlight-detect/data)
- **Dữ liệu gán nhãn rổ (Roboflow)**: [Basketball Hoop Dataset](https://universe.roboflow.com/ojas-ecz6e/basketball-hoop-igupq)
- **Trọng số Mô hình (Trained Models)**: [Kaggle - Model Basketball Highlight Detect](https://www.kaggle.com/datasets/nonamenood/model-basketball-hightlight-detect)
- **Notebook Huấn luyện (Training Evidence)**: [Kaggle Code - Basketball Dataset](https://www.kaggle.com/code/nonamenood/basketball-dataset/edit)

## ⚙️ Cách thức hoạt động (How it works)
Hệ thống xử lý video đầu vào thông qua một **Pipeline 3 bước**:

1. **Phát hiện rổ (Rim Detection)**: Sử dụng mô hình nhận diện vật thể **YOLOv8** (`yolov8n.pt`) để quét qua các khung hình (frame) của video nhằm phát hiện vị trí của rổ bóng rổ.
2. **Đánh giá khả năng ghi điểm (Score Classifier)**: Cắt các khung hình chứa rổ bóng rổ (crops) lấy từ bước 1 và đưa vào mô hình phân loại hình ảnh (Image Classifier) - như **ResNet** - để dự đoán xác suất khung hình đó có phải là một pha bóng ghi điểm hay không (Point vs No Point).
3. **Trích xuất Highlight (Highlight Extraction)**: Tổng hợp xác suất ghi điểm từ bước 2 cùng với cường độ âm thanh của video (Audio Weight) để xác định các "đỉnh" (peaks) cao trào. Hệ thống sẽ tự động cắt các đoạn clip ngắn xung quanh những điểm cao trào này và ghép lại thành một video highlight hoàn chỉnh.

## 🗂 Cấu trúc thư mục (Project Structure)
- `app.py`: Giao diện Web (UI) sử dụng **Streamlit**.
- `pipeline.py` & `make_highlights.py`: Chứa logic xử lý chính của quy trình, bao gồm chạy inference và thuật toán cắt, ghép video.
- `train_detector.py` & `train_classifier.py`: Script dùng để huấn luyện mô hình.
- `config.yaml`: File cấu hình các tham số huấn luyện.
- `checkpoints/` & `runs/`: Thư mục lưu trữ các trọng số (weights) của các mô hình đã được huấn luyện.

---

## ☁️ Hướng dẫn chạy trên Kaggle (Khuyên dùng để Train)

Bạn có thể tham khảo [Notebook Huấn luyện gốc](https://www.kaggle.com/code/nonamenood/basketball-dataset/edit) hoặc tự setup theo các bước sau:

### Bước 1: Khởi tạo Notebook & Thêm Dữ liệu
1. Tạo một Notebook mới trên [Kaggle](https://www.kaggle.com/).
2. Trong Notebook, chọn **Add Data** ở thanh menu bên phải.
3. Dán link [Dữ liệu ảnh Raw](https://www.kaggle.com/datasets/nonamenood/dataset-basketball-highlight-detect/data) và nhấn **Add** (nút dấu +) để đính kèm dataset vào Notebook. 
*(Dữ liệu sẽ nằm ở đường dẫn `/kaggle/input/dataset-basketball-highlight-detect/`)*

### Bước 2: Tải Mã nguồn & Thiết lập Môi trường
Tạo một ô code (Code Cell) và chạy các lệnh sau:

```python
# Clone mã nguồn từ Github (Thay link bằng link thật của repo)
!git clone https://github.com/your-username/Basketball-Highlight-Detect.git
%cd Basketball-Highlight-Detect

# Tạo thư mục data và copy dữ liệu từ Kaggle input vào project
!mkdir -p data
!cp -r /kaggle/input/dataset-basketball-highlight-detect/crops data/
!cp -r /kaggle/input/dataset-basketball-highlight-detect/yolo_rim data/

# Cài đặt thư viện cần thiết
!pip install -r requirements.txt
```

### Bước 3: Tiến hành Huấn luyện (Training)

**Train YOLO (Nhận diện rổ):**
```python
!python train_detector.py
```

**Train Classifier (Nhận diện pha ghi điểm):**
```python
!python train_classifier.py
```

---

## 💻 Hướng dẫn chạy trên Máy Cá Nhân (Local - Dùng để chạy Inference Web)

### 1. Cài đặt Môi trường
1. Clone mã nguồn về máy:
```bash
git clone https://github.com/your-username/Basketball-Highlight-Detect.git
cd Basketball-Highlight-Detect
```
2. Cài đặt các thư viện yêu cầu:
```bash
pip install -r requirements.txt
```

### 2. Thiết lập Model
Bạn không cần phải tự train lại, hãy tải trực tiếp các model đã được train sẵn tại [Model Basketball Highlight Detect](https://www.kaggle.com/datasets/nonamenood/model-basketball-hightlight-detect).
1. Tải về 2 file: `trained_models.zip` (Classifier) và `yolo_models.zip` (YOLO).
2. Giải nén `trained_models.zip` vào thư mục `checkpoints/`.
3. Giải nén `yolo_models.zip` vào thư mục `runs/`.

### 3. Chạy Ứng dụng (Streamlit)
1. Khởi chạy giao diện bằng lệnh:
```bash
streamlit run app.py
```
2. Trình duyệt sẽ mở ra trang web ứng dụng (Mặc định ở `http://localhost:8501`).
3. **Cách dùng**:
   - Tải lên một đoạn video bóng rổ (`.mp4`, `.avi`, `.mov`).
   - Nhấn **🚀 Chạy Pipeline** và chờ hệ thống phân tích để nhận video highlight thành quả.
