import streamlit as st
import os
from pathlib import Path
import cv2

# Import pipeline modules
from infer_rim_video import run_rim_inference
from score_rim_crops import run_score_crops
from make_highlights import run_highlight_extraction

st.set_page_config(page_title="Basketball Highlight Detection", page_icon="🏀", layout="wide")

st.title("🏀 Basketball Highlight Detection Pipeline")
st.write("Tải lên một video bóng rổ để hệ thống tự động tìm và cắt các pha ghi điểm (Highlights).")

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Settings
with st.sidebar:
    st.header("Cài đặt Pipeline (MVP)")
    # test_mode = st.checkbox("Chế độ Test nhanh (chỉ chạy 500 frames đầu)", value=True, help="Bỏ chọn để xử lý toàn bộ video.")
    max_frames = None
    rim_threshold = st.slider("Ngưỡng nhận diện rổ (Rim Det. Thresh)", min_value=0.05, max_value=0.8, value=0.15, step=0.05, help="Giảm mức này nếu hệ thống không tìm thấy rổ bóng nào.")
    peak_threshold = st.slider("Ngưỡng điểm tin cậy (Peak Threshold)", min_value=0.1, max_value=0.9, value=0.5, step=0.05, help="Giảm mức này nếu video không tìm thấy highlight nào.")
    audio_weight = st.slider("Trọng số Âm thanh (Audio Weight)", min_value=0.0, max_value=2.0, value=1.0, step=0.1, help="Kết hợp độ lớn âm thanh để tìm highlight. Đặt về 0 để tắt.")
    
uploaded_file = st.file_uploader("Chọn một video clip (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file safely to avoid PermissionError on Windows
    video_path = UPLOAD_DIR / uploaded_file.name
    if not video_path.exists() or video_path.stat().st_size != uploaded_file.size:
        try:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except PermissionError:
            pass # File is likely being used by Streamlit's video player, which is fine

        
    st.video(str(video_path))
    
    if st.button("🚀 Chạy Pipeline", type="primary"):
        st.markdown("---")
        
        # Get video FPS
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0: fps = 25.0
        
        # 1. Rim Detection
        st.subheader("1. Phát hiện rổ (Rim Detection) bằng YOLO")
        progress_bar_1 = st.progress(0)
        status_text_1 = st.empty()
        
        def update_progress_1(current, total):
            if current % 100 == 0 or current == total:
                progress = min(current / total, 1.0) if total > 0 else 0
                progress_bar_1.progress(progress)
                status_text_1.text(f"Đang xử lý: {current}/{total} frames")
            
        csv_rim_path = run_rim_inference(
            video_path=str(video_path), 
            output_dir=str(OUTPUT_DIR / "rim_inference"), 
            max_frames=max_frames,
            progress_callback=update_progress_1,
            conf_thres=rim_threshold
        )
        st.success(f"Hoàn thành phát hiện rổ. Data lưu tại: {csv_rim_path}")
        
        # 2. Score Crops
        st.subheader("2. Đánh giá khả năng ghi điểm (ResNet Classifier)")
        progress_bar_2 = st.progress(0)
        status_text_2 = st.empty()
        
        def update_progress_2(current, total):
            if current % 100 == 0 or current == total:
                progress = min(current / total, 1.0) if total > 0 else 0
                progress_bar_2.progress(progress)
                status_text_2.text(f"Đang chấm điểm: {current}/{total} crops")
            
        csv_score_path = run_score_crops(
            csv_input=csv_rim_path, 
            output_dir=str(OUTPUT_DIR / "score_inference"),
            progress_callback=update_progress_2
        )
        st.success(f"Hoàn thành chấm điểm. File điểm số: {csv_score_path}")
        
        # 3. Extract Highlights & Build Video
        st.subheader("3. Trích xuất khoảnh khắc & Render Video")
        progress_bar_3 = st.progress(0)
        status_text_3 = st.empty()
        
        def update_progress_3(current, total):
            progress = min(current / total, 1.0) if total > 0 else 0
            progress_bar_3.progress(progress)
            status_text_3.text(f"Đang render clip highlight: {current}/{total}")
            
        final_video_path = run_highlight_extraction(
            video_path=str(video_path),
            csv_path=csv_score_path,
            output_dir=str(OUTPUT_DIR),
            fps=fps,
            height=peak_threshold,
            audio_weight=audio_weight,
            progress_callback=update_progress_3
        )
        
        if final_video_path and Path(final_video_path).exists():
            st.success("Tạo video Highlight thành công! 🎉")
            st.markdown("---")
            st.subheader("🎬 Kết quả (Highlight Video)")
            
            st.video(final_video_path)
            
            with open(final_video_path, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Tải xuống Video Highlight",
                    data=file,
                    file_name="final_highlight.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("Không tìm thấy khoảnh khắc ghi điểm nào đủ rõ ràng để tạo highlight, hoặc video quá ngắn.")
