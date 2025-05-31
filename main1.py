import time
import streamlit as st
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from shapely.geometry import box as shapely_box

# ============ THIẾT LẬP ==============
# Bảng màu cho các lớp lúa khác nhau
# Định nghĩa mapping màu cho các class
color_class_mapping = {
    0: (246, 202, 0),  # Màu vàng
    1: (214, 39, 104),  # Màu hồng
    2: (65, 105, 225),  # Xanh dương (đã điều chỉnh: Royal Blue nhẹ hơn)
    3: (34, 139, 34),  # Xanh lá (đã điều chỉnh: Forest Green thay vì xanh lá sáng)
    4: (220, 20, 60),  # Đỏ (đã điều chỉnh: Crimson thay vì đỏ tươi)
    5: (128, 0, 128),  # Màu tím
    6: (255, 140, 0),  # Màu cam (đã điều chỉnh: Dark Orange)
    7: (32, 178, 170)  # Xanh ngọc (đã điều chỉnh: Light Sea Green thay vì Cyan)
}

# Chuyển đổi màu RGB sang hex cho matplotlib
color_hex_mapping = {
    0: '#F6CA00',  # Vàng (246, 202, 0)
    1: '#D62768',  # Hồng (214, 39, 104)
    2: '#4169E1',  # Royal Blue (65, 105, 225)
    3: '#228B22',  # Forest Green (34, 139, 34)
    4: '#DC143C',  # Crimson (220, 20, 60)
    5: '#800080',  # Tím (128, 0, 128)
    6: '#FF8C00',  # Dark Orange (255, 140, 0)
    7: '#20B2AA'   # Light Sea Green (32, 178, 170)
}

class_names = ['ANP3.2', 'F5.54', 'F5.62', 'G13.II', 'G17.III', 'G18-I', 'G3.III', 'G7.III']

# Cấu hình trang
st.set_page_config(
    page_title="Ứng Dụng Phân Loại Lúa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh để tạo giao diện đẹp hơn
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A8A;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .result-container {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .stSelectbox label, .stSlider label {
        font-weight: 500;
        color: #4B5563;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# ============ THANH BÊN ==============
with st.sidebar:
    st.markdown('<div class="sidebar-header">Công Cụ Phân Loại Hạt Giống Lúa</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Lựa chọn phương án với giao diện đẹp hơn
    st.markdown('<div class="sidebar-header">Cấu Hình</div>', unsafe_allow_html=True)
    phuong_an = st.radio(
        "Chọn Phương Án",
        ["Phương án 1", "Phương án 2"],
        index=0,
        help="Chọn giữa hai phương pháp phân loại khác nhau"
    )

    # Lựa chọn mô hình dựa trên phương án
    if phuong_an == "Phương án 1":
        mix_model_name = st.selectbox(
            "Mô Hình YOLO & Faster R-CNN",
            ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv11n","YOLOv11s", "YOLOv11m","YOLOv11l","Faster R-CNN 50","Faster R-CNN 101","YOLOv11n Update"],
            help="Chọn mô hình YOLO cho phát hiện và phân loại tích hợp"
        )
    else:
        classify_model_name = st.selectbox(
            "Mô Hình Phân Loại",
            ["EfficientNetB3", "InceptionV3", "ResNet50", "ViT"],
            help="Chọn mô hình để phân loại lúa"
        )


    confidence_threshold = st.slider(
        "Ngưỡng Độ Tin Cậy",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Điều chỉnh ngưỡng độ tin cậy cho việc phân loại"
    )

    st.markdown("---")
    st.markdown('<div class="footer">Phát triển cho Phân Loại Lúa Giống</div>', unsafe_allow_html=True)

# ============ NỘI DUNG CHÍNH ==============
# Tiêu đề
st.markdown('<div class="main-header">Hệ Thống Phân Loại Lúa</div>', unsafe_allow_html=True)

# Giới thiệu
with st.expander("Giới Thiệu Về Ứng Dụng", expanded=False):
    st.markdown("""
    Ứng dụng này sử dụng thị giác máy tính và học sâu để phân loại hạt lúa giống thành một trong tám lớp sau:

    - **ANP3.2**
    - **F5.54**
    - **F5.62**
    - **G13.II**
    - **G17.III**
    - **G18-I**
    - **G3.III**
    - **G7.III**

    Hệ thống cung cấp hai phương án:
    1. **Phương án 1**: Phát hiện và phân loại tích hợp với các mô hình YOLO, Faster R-CNN
    2. **Phương án 2**: Quy trình hai giai đoạn với phát hiện bằng YOLO sau đó là phân loại bằng mô hình riêng
    """)

# Phần tải lên với giao diện đẹp hơn
st.markdown('<div class="sub-header">Tải Lên Hình Ảnh</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Tải lên hình ảnh hạt lúa", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)


# ============ ĐỊNH NGHĨA HÀM ==============
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def get_config_faster_r_cnn(weights_path ,num_of_class,threshold):
        """
        Cấu hình cho model Detectron2
        """
        if "101" in weights_path:
            config_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        else:
            config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

        cfg = get_cfg()

        # Cấu hình cơ bản
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = weights_path

        # Thiết lập thiết bị và hiệu suất
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #cfg.SOLVER.AMP.ENABLED = True  # Sử dụng mixed precision

        # Cấu hình neo (anchors)
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

        # Cấu hình đầu ra và threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class + 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Ngưỡng tin cậy cho dự đoán
        cfg.TEST.DETECTIONS_PER_IMAGE = 100  # Số lượng phát hiện tối đa mỗi ảnh

        # Loại bỏ các cấu hình không cần thiết cho dự đoán
        # if hasattr(cfg, 'SOLVER'):
        #     cfg.SOLVER.IMS_PER_BATCH = 1  # Cho dự đoán, thường chỉ cần batch size = 1
        return cfg
def xywh_to_xyxy(box_xywh):

    x_center, y_center, width, height = box_xywh
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]
def calculate_iou(box1, box2):

    # Chuyển đổi sang định dạng shapely box: (minx, miny, maxx, maxy)
    box1_shapely = shapely_box(box1[0], box1[1], box1[2], box1[3])
    box2_shapely = shapely_box(box2[0], box2[1], box2[2], box2[3])

    # Tính diện tích giao và hợp
    intersection_area = box1_shapely.intersection(box2_shapely).area
    union_area = box1_shapely.union(box2_shapely).area

    # Tính IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def find_unclassified_boxes(detect_boxes, predict_boxes, iou_threshold=0.5):
    # Đánh dấu các box đã match

    detect_boxes_xyxy = [xywh_to_xyxy(box) for box in detect_boxes]

    matched = [False] * len(detect_boxes)

    # Kiểm tra từng box YOLO
    for i, detect_box in enumerate(detect_boxes_xyxy):
        for predict_box in predict_boxes:
            iou = calculate_iou(detect_box, predict_box)
            if iou >= iou_threshold:
                matched[i] = True
                break

    # Lấy các box chưa được match
    unclassified_boxes = [detect_boxes[i] for i in range(len(detect_boxes)) if not matched[i]]
    unclassified_indices = [i for i in range(len(detect_boxes)) if not matched[i]]

    return unclassified_boxes, unclassified_indices
def load_classify_model(model_name, model_path, num_classes=8):
    if model_name == "InceptionV3":
        model = torch.load(model_path, weights_only=False, map_location='cpu')
        model.eval()

        return model
    if model_name == "ResNet50":
        model = torch.load(model_path, weights_only=False , map_location='cpu')
        model.eval()

        return model
    if model_name == "ViT":
        model = torch.load(model_path, weights_only=False,map_location='cpu')
        model.eval()

        return model
    if model_name == "EfficientNetB3":
        model = torch.load(model_path, weights_only=False , map_location='cpu')
        model.eval()

        return model
    st.warning("Mô hình chưa được hỗ trợ.")
    return None
def prepare_crop_50(crop):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    return transform(crop).unsqueeze(0)

def prepare_crop_ViT(crop):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    return transform(crop).unsqueeze(0)
def prepare_crop_B3(crop):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    return transform(crop).unsqueeze(0)


def prepare_crop_V3(crop):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    return transform(crop).unsqueeze(0)


def crop_objects(image, boxes):
    crops = []
    image = np.array(image)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        crops.append(crop)
    return crops


def classify_crops_B3(model, crop_list):
    predict = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for crop in crop_list:
            input_tensor = prepare_crop_B3(crop)
            output = model(input_tensor)

            # Calculate probabilities
            probs = torch.nn.functional.softmax(output, dim=1)

            # Get predicted class and confidence
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

            # Combine into a dict
            predict.append({
                "class": pred_class,
                "confidence": confidence
            })

    return predict
def classify_crops_V3(model, crop_list):
    predict = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for crop in crop_list:
            # Assuming prepare_crop_ViT exists for Vision Transformer
            input_tensor = prepare_crop_ViT(crop)
            output = model(input_tensor)

            # Calculate probabilities
            probs = torch.nn.functional.softmax(output, dim=1)

            # Get predicted class and confidence
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

            predict.append({
                "class": pred_class,
                "confidence": confidence
            })

    return predict


def classify_crops_50(model, crop_list):
    predict = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for crop in crop_list:
            # Assuming prepare_crop_50 exists for ResNet50
            input_tensor = prepare_crop_50(crop)
            output = model(input_tensor)

            # Calculate probabilities
            probs = torch.nn.functional.softmax(output, dim=1)

            # Get predicted class and confidence
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

            predict.append({
                "class": pred_class,
                "confidence": confidence
            })

    return predict

def classify_crops_ViT(model, crop_list):
    predict = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for crop in crop_list:
            # Assuming prepare_crop_ViT exists for Vision Transformer
            input_tensor = prepare_crop_ViT(crop)
            output = model(input_tensor)

            # Calculate probabilities
            probs = torch.nn.functional.softmax(output, dim=1)

            # Get predicted class and confidence
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

            predict.append({
                "class": pred_class,
                "confidence": confidence
            })

    return predict

def draw_boxes(image, boxes, labels=None, confidence=None,
               line_width=None, font_size=None, font_path="./Font/Arial.ttf", pil=True):
    if not pil:
        raise NotImplementedError("Hiện chỉ hỗ trợ PIL để vẽ theo cấu hình này.")

    # Màu mặc định nếu class không có trong mapping
    default_color = (64, 97, 190)

    # Chuyển sang ảnh PIL nếu chưa phải
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Lấy kích thước ảnh
    img_width, img_height = image.size
    img_size = img_width + img_height

    # Áp dụng scaling cho line_width và font_size nếu không được chỉ định
    if line_width is None:
        line_width = max(round(img_size * 0.0015), 2)  # Scaling dựa trên kích thước ảnh

    if font_size is None:
        font_size = max(round(img_size * 0.015), 12)  # Scaling dựa trên kích thước ảnh

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default(font_size)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Xác định màu dựa trên class
        label_idx = labels[i] if labels and i < len(labels) else 0

        # Đảm bảo label_idx là số nguyên để sử dụng làm chỉ mục trong mapping
        if isinstance(label_idx, int) and label_idx in color_class_mapping:
            box_color = color_class_mapping[label_idx]
        else:
            # Thử chuyển đổi nếu label_idx không phải số nguyên
            try:
                class_id = int(label_idx)
                box_color = color_class_mapping.get(class_id, default_color)
            except (ValueError, TypeError):
                box_color = default_color

        if confidence is None:
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=line_width)
        else:
            if confidence[i] == 0:
                draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=line_width)
                continue
            else:
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        if labels and confidence and i < len(labels) and i < len(confidence):
            conf = confidence[i]
            label_idx = labels[i]

            if isinstance(label_idx, int):
                label = class_names[label_idx] if label_idx < len(class_names) else "Unknown"
            else:
                label = label_idx

            conf_str = f"{conf:.2f}" if conf != "" else ""
            label_text = f"{label}  {conf_str}"

            text_size = draw.textbbox((x1, y1), label_text, font=font)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]

            # Scaling cho background của label text
            padding_x = max(round(img_size * 0.001), 1)  # Padding theo chiều ngang
            padding_y = max(round(img_size * 0.001), 1)  # Padding theo chiều dọc

            # Vẽ background cho text với cùng màu như bounding box
            draw.rectangle([
                x1,
                y1 - text_h - padding_y * 2,
                x1 + text_w + padding_x * 2,
                y1
            ], fill=box_color)

            # Vẽ text với màu trắng (giữ nguyên)
            draw.text(
                (x1 + padding_x, y1 - text_h - 3 * padding_y),
                label_text,
                fill="white",
                font=font
            )

    return np.array(image)
def plot_colored_bar_chart(data, title="Phân Bố Các Loại Lúa"):
    """Tạo biểu đồ cột với màu sắc tùy chỉnh cho từng loại lúa"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

    # Sắp xếp dữ liệu theo giá trị giảm dần
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # Tạo danh sách màu sắc cho từng loại
    colors = []
    for class_name in sorted_data.keys():
        if class_name == "Chưa Phân Loại" or class_name == "Không Xác Định" or class_name == "Không phân loại":
            colors.append('#FF0000')  # Màu đỏ cho các loại không phân loại được
        elif class_name == 'ANP3.2':
            colors.append(color_hex_mapping[0])
        elif class_name == 'F5.54':
            colors.append(color_hex_mapping[1])
        elif class_name == 'F5.62':
            colors.append(color_hex_mapping[2])
        elif class_name == 'G13.II':
            colors.append(color_hex_mapping[3])
        elif class_name == 'G17.III':
            colors.append(color_hex_mapping[4])
        elif class_name == 'G18-I':
            colors.append(color_hex_mapping[5])
        elif class_name == 'G3.III':
            colors.append(color_hex_mapping[6])
        elif class_name == 'G7.III':
            colors.append(color_hex_mapping[7])
        else:
            colors.append('#808080')  # Màu xám cho các loại khác không xác định

    # Vẽ biểu đồ
    bars = ax.bar(range(len(sorted_data)), list(sorted_data.values()), color=colors)

    # Thêm nhãn và tiêu đề
    ax.set_xticks(range(len(sorted_data)))
    ax.set_xticklabels(list(sorted_data.keys()), rotation=45, ha='right')  # Xoay nhãn 45 độ để dễ đọc
    ax.set_ylabel('Số Lượng')
    ax.set_title(title)

    # Thêm giá trị lên đầu mỗi cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    return fig

def plot_confidence_histogram(confidences, bins=10, title="Phân Phối Độ Tin Cậy"):
    """Tạo biểu đồ histogram cho độ tin cậy"""
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

    # Tạo histogram
    n, bins, patches = ax.hist(confidences, bins=bins, range=(0, 1), color='#1E3A8A', alpha=0.7)

    # Thêm nhãn và tiêu đề
    ax.set_xlabel('Độ Tin Cậy')
    ax.set_ylabel('Số Lượng')
    ax.set_title(title)

    # Thêm giá trị lên đầu mỗi cột
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(n, bin_centers):
        if count > 0:
            ax.text(x, count + 0.1, f'{int(count)}', ha='center', va='bottom')

    plt.tight_layout()
    return fig

# ============ LOGIC CHÍNH ==============
image = None
if uploaded_file:
    # Hiển thị hình ảnh đã tải lên với giao diện đẹp hơn
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown('<div class="sub-header">Hình Ảnh Đã Tải Lên</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Hình Ảnh Đã Tải Lên", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Nút xử lý với giao diện đẹp hơn
    st.markdown('<div class="card">', unsafe_allow_html=True)
    process_button = st.button("Xử Lý Hình Ảnh", help="Chạy phát hiện và phân loại trên hình ảnh đã tải lên")
    st.markdown('</div>', unsafe_allow_html=True)

    if process_button:
        # Hiển thị chỉ báo xử lý
        with st.spinner("Đang xử lý hình ảnh..."):
            progress_bar = st.progress(0)

            # Phương án 1: Quy trình hai giai đoạn
            if phuong_an == "Phương án 2":
                st.markdown('<div class="sub-header">Kết Quả - Quy Trình Hai Giai Đoạn</div>', unsafe_allow_html=True)

                # Tải mô hình
                progress_bar.progress(20)
                yolo_model = YOLO("./Phuong an 2/best_11s.pt")

                if classify_model_name[0] == "E":
                    model_path = "./Phuong an 1/efficientNet.pth"
                elif classify_model_name[0] == "I":
                    model_path = "./Phuong an 1/inception_V3.pth"
                elif classify_model_name[0] == "R":
                    model_path = "./Phuong an 1/ResNet_50.pth"
                else:
                    model_path = "./Phuong an 1/ViT.pth"

                classify_model = load_classify_model(classify_model_name, model_path)

                # Chạy phát hiện YOLO
                progress_bar.progress(40)
                prediction_timer = Timer()
                prediction_timer.start()
                results = yolo_model.predict(image, conf=0.6)[0]

                boxes = results.boxes.xyxy.cpu().numpy()

                # Cắt và phân loại
                progress_bar.progress(60)
                crops = crop_objects(image, boxes)

                # Chạy phân loại dựa trên mô hình đã chọn
                progress_bar.progress(80)
                if classify_model_name == "EfficientNetB3":
                    predicted = classify_crops_B3(classify_model, crops)
                elif classify_model_name == "InceptionV3":
                    predicted = classify_crops_V3(classify_model, crops)
                elif classify_model_name == "ResNet50":
                    predicted = classify_crops_50(classify_model, crops)
                elif classify_model_name == "ViT":
                    predicted = classify_crops_ViT(classify_model, crops)

                prediction_timer.stop()

                # Xử lý kết quả
                labels = []
                confs = []

                num_missing = 0

                for predict in predicted:
                    if predict['confidence'] >= confidence_threshold:
                        labels.append(predict['class'])
                        confs.append(predict['confidence'])
                    else:
                        confs.append(0)
                        labels.append(1000)
                        num_missing += 1

                prediction_time = prediction_timer.get_elapsed_time()
                # Hiển thị kết quả
                progress_bar.progress(100)

                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                image2 = image.copy()
                with col1:
                    st.markdown(
                        '<div style="text-align:center; font-weight:600; margin-bottom:10px;">Hình Ảnh Gốc</div>',
                        unsafe_allow_html=True)
                    st.image(image, use_container_width=True)

                with col2:
                    st.markdown(
                        '<div style="text-align:center; font-weight:600; margin-bottom:10px;">Phát Hiện YOLO</div>',
                        unsafe_allow_html=True)
                    st.image(draw_boxes(image, boxes), use_container_width=True)

                with col3:
                    st.markdown(
                        f'<div style="text-align:center; font-weight:600; margin-bottom:10px;">Phân Loại ({classify_model_name})</div>',
                        unsafe_allow_html=True)
                    st.image(draw_boxes(image2, boxes, labels, confs), use_container_width=True)

                st.markdown(
                        f'<div style="text-align:center; font-weight:600; margin-top:10px;">Thời gian dự đoán: {prediction_time:.4f} giây</div>',
                        unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Hiển thị thống kê
                st.markdown('<div class="sub-header">Thống Kê Phân Loại</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)

                # Đếm số lần xuất hiện của mỗi lớp
                class_counts = {}
                for label in labels:
                    class_name = class_names[label] if label < len(class_names) else "Không phân loại"
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1

                if num_missing > 0:
                    class_counts["Không phân loại"] = num_missing

                # Hiển thị dưới dạng biểu đồ thanh ngang với màu sắc tùy chỉnh
                if class_counts:
                    # Tạo biểu đồ cột với màu sắc tùy chỉnh
                    fig = plot_colored_bar_chart(class_counts)
                    st.pyplot(fig)

                    # Hiển thị dưới dạng bảng
                    st.markdown("### Kết Quả Chi Tiết")
                    stats_data = {"Loại": list(class_counts.keys()), "Số Lượng": list(class_counts.values())}
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

                    # Hiển thị biểu đồ tròn
                    st.markdown("### Tỷ Lệ Các Loại Lúa")
                    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                    colors_pie = []
                    for class_name in class_counts.keys():
                        if class_name == "Không phân loại":
                            colors_pie.append('#FF0000')
                        else:
                            idx = class_names.index(class_name)
                            colors_pie.append(color_hex_mapping[idx])

                    wedges, texts, autotexts = ax_pie.pie(
                        list(class_counts.values()),
                        labels=list(class_counts.keys()),
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors_pie
                    )
                    # Làm cho văn bản dễ đọc hơn
                    for text in texts:
                        text.set_fontsize(12)
                    for autotext in autotexts:
                        autotext.set_fontsize(12)
                        autotext.set_fontweight('bold')

                    ax_pie.set_title('Tỷ Lệ Các Loại Lúa', fontsize=14)
                    ax_pie.axis('equal')  # Đảm bảo biểu đồ tròn là hình tròn
                    st.pyplot(fig_pie)
                else:
                    st.info("Không phát hiện hạt lúa nào trong hình ảnh.")

                st.markdown('</div>', unsafe_allow_html=True)

            # Phương án 2: mô hình tích hợp (YOLO, FASTER R-CNN)
            else:
                def model_to_weight_path(model_name, mode="YOLO"):
                    if mode != "YOLO":
                        suffix = model_name.replace("Faster R-CNN ", "")
                        return f"model_final_{suffix}.pth"
                    else:
                        # Lấy số và chữ cái cuối từ tên mô hình (ví dụ: '8n' từ 'yolov8n')
                        suffix = model_name.replace("YOLOv", "")
                        return f"best_{suffix}.pt"

                # Tải mô hình
                progress_bar.progress(30)

                if "Faster R-CNN" in mix_model_name:
                    st.markdown('<div class="sub-header">Kết Quả - Faster R-CNN</div>', unsafe_allow_html=True)

                    # Tải mô hình Faster R-CNN
                    progress_bar.progress(30)

                    backbone = "resnet50" if "ResNet50" in mix_model_name else "ResNet101"

                    model_name = model_to_weight_path(mix_model_name, mode="Faster R-CNN",)

                    weight_path =  f"./Phuong an 2/{model_name}"

                    cfg = get_config_faster_r_cnn(weights_path=weight_path, num_of_class=len(class_names), threshold = confidence_threshold)

                    # Chuyển sang numpy array
                    img_array = np.array(image)

                    img = img_array

                    img_BGR = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    predictor = DefaultPredictor(cfg)

                    my_metadata = MetadataCatalog.get("my_dataset")

                    my_metadata.thing_classes = ["Rice", "ANP3.2", "F5.54", "F5.62", "G13.II", "G17.III", "G18-I",
                                                 "G3.III", "G7.III"]


                    # Chạy phát hiện và phân loại với Faster R-CNN
                    progress_bar.progress(60)

                    yolo_model = YOLO("./Phuong an 2/best_11s.pt")
                    yolo_detections = yolo_model.predict(image,conf = 0.6)
                    gt_of_object = len(yolo_detections[0].boxes)

                    detect_boxes = [box.xywh[0].tolist() for box in yolo_detections[0].boxes]

                    # Dự đoán
                    prediction_timer = Timer()
                    prediction_timer.start()
                    outputs = predictor(img_BGR)
                    prediction_timer.stop()
                    prediction_time = prediction_timer.get_elapsed_time()

                    # Tiến hành Visualize kết quả
                    v_pred = Visualizer(img, metadata=my_metadata)
                    pred_vis = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))


                    instances = outputs["instances"].to("cpu")

                    boxes = instances.pred_boxes.tensor.numpy().tolist()
                    labels = (instances.pred_classes.numpy() - 1).tolist()
                    confs = instances.scores.numpy().tolist()

                    # Tính số lượng vật thể không phân loại được
                    predict_of_object = gt_of_object - len(boxes)

                    # Tìm vật thể không phân loại và vẽ nó lên hình
                    missing_box = find_unclassified_boxes(detect_boxes, boxes ,0.5)[0]

                    missing_box_xyxy = [xywh_to_xyxy(box) for box in missing_box]

                    result_img = draw_boxes(img,missing_box_xyxy)
                    result_img = draw_boxes(result_img, boxes, labels, confs)

                    # Cách này sử dụng mô-đun Visualizer có sẳn của Detectron2
                    # result_img = pred_vis.get_image()
                    #
                    # result_img = draw_boxes(result_img,missing_box_xyxy)

                    # Hiển thị kết quả
                    progress_bar.progress(100)

                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            '<div style="text-align:center; font-weight:600; margin-bottom:10px;">Hình Ảnh Gốc</div>',
                            unsafe_allow_html=True)
                        st.image(image, use_container_width=True)

                    with col2:
                        st.markdown(
                            f'<div style="text-align:center; font-weight:600; margin-bottom:10px;">Kết Quả {mix_model_name}</div>',
                            unsafe_allow_html=True)
                        st.image(result_img, use_container_width=True)

                    st.markdown(
                        f'<div style="text-align:center; font-weight:600; margin-top:10px;">Thời gian dự đoán: {prediction_time:.4f} giây</div>',
                        unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Hiển thị thống kê phát hiện
                    st.markdown('<div class="sub-header">Thống Kê Phát Hiện</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    if len(boxes) > 0:
                        # Đếm số lần xuất hiện của mỗi lớp
                        class_counts = {}
                        for label in labels:
                            class_name = class_names[label] if label < len(class_names) else "Không phân loại"
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1

                        if predict_of_object > 0:
                            class_counts["Không phân loại"] = predict_of_object

                        # Tạo biểu đồ cột với màu sắc tùy chỉnh
                        fig = plot_colored_bar_chart(class_counts)
                        st.pyplot(fig)

                        # Hiển thị dưới dạng bảng
                        st.markdown("### Kết Quả Chi Tiết")
                        stats_data = {"Loại": list(class_counts.keys()), "Số Lượng": list(class_counts.values())}
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

                        if predict_of_object > 0:
                            st.markdown("""
                            > **ℹ️ Ghi chú:**  
                            > Các đối tượng được gắn nhãn **"Không phân loại"** trong trường hợp này là những đối tượng được phát hiện bởi mô hình phát hiện **YOLOv11s** nhưng không được tìm thấy bởi mô hình này 
                            """)
                        # Hiển thị biểu đồ tròn
                        st.markdown("### Tỷ Lệ Các Loại Lúa")
                        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                        colors_pie = []
                        for class_name in class_counts.keys():
                            if class_name == "Không phân loại":
                                colors_pie.append('#FF0000')
                            else:
                                class_idx = class_names.index(class_name)
                                colors_pie.append(color_hex_mapping.get(class_idx, '#808080'))

                        wedges, texts, autotexts = ax_pie.pie(
                            list(class_counts.values()),
                            labels=list(class_counts.keys()),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors_pie
                        )
                        # Làm cho văn bản dễ đọc hơn

                        for text in texts:
                            text.set_fontsize(12)
                        for autotext in autotexts:
                            autotext.set_fontsize(12)
                            autotext.set_fontweight('bold')

                        ax_pie.set_title('Tỷ Lệ Các Loại Lúa', fontsize=14)
                        ax_pie.axis('equal')  # Đảm bảo biểu đồ tròn là hình tròn
                        st.pyplot(fig_pie)

                        # Hiển thị thông tin độ tin cậy
                        st.markdown("### Phân Phối Độ Tin Cậy")
                        st.write(f"Độ Tin Cậy Trung Bình: {np.mean(confs):.2f}")
                        st.write(f"Độ Tin Cậy Thấp Nhất: {np.min(confs):.2f}")
                        st.write(f"Độ Tin Cậy Cao Nhất: {np.max(confs):.2f}")

                        # Biểu đồ phân phối độ tin cậy
                        fig_conf = plot_confidence_histogram(confs)
                        st.pyplot(fig_conf)
                    else:
                        st.info("Không phát hiện hạt lúa nào trong hình ảnh.")

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub-header">Kết Quả - YOLO Tích Hợp</div>', unsafe_allow_html=True)
                    # Ví dụ sử dụng
                    model_name = model_to_weight_path(mix_model_name)
                    model_path = f"./Phuong an 2/{model_name}"
                    yolo_model = YOLO(model_path)

                    # Chạy dự đoán
                    progress_bar.progress(70)
                    prediction_timer = Timer()
                    prediction_timer.start()
                    results = yolo_model.predict(image, conf=confidence_threshold)
                    prediction_timer.stop()
                    prediction_time = prediction_timer.get_elapsed_time()
                    result_img = results[0].plot()
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    # Hiển thị kết quả
                    progress_bar.progress(100)

                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            '<div style="text-align:center; font-weight:600; margin-bottom:10px;">Hình Ảnh Gốc</div>',
                            unsafe_allow_html=True)
                        st.image(image, use_container_width=True)

                    with col2:
                        st.markdown(
                            f'<div style="text-align:center; font-weight:600; margin-bottom:10px;">Kết Quả {mix_model_name}</div>',
                            unsafe_allow_html=True)
                        st.image(result_img, use_container_width=True)

                    # Hiển thị thời gian dự đoán
                    st.markdown(
                        f'<div style="text-align:center; font-weight:600; margin-top:10px;">Thời gian dự đoán: {prediction_time:.4f} giây</div>',
                        unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Hiển thị thống kê phát hiện
                    st.markdown('<div class="sub-header">Thống Kê Phát Hiện</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    # Trích xuất thông tin lớp từ kết quả
                    if len(results[0].boxes) > 0:
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                        class_counts = {}

                        for class_id in class_ids:
                            class_name = class_names[class_id]
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1

                        # Tạo biểu đồ cột với màu sắc tùy chỉnh
                        fig = plot_colored_bar_chart(class_counts)
                        st.pyplot(fig)

                        # Hiển thị dưới dạng bảng
                        st.markdown("### Kết Quả Chi Tiết")
                        stats_data = {"Loại": list(class_counts.keys()), "Số Lượng": list(class_counts.values())}
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

                        # Hiển thị biểu đồ tròn
                        st.markdown("### Tỷ Lệ Các Loại Lúa")
                        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
                        colors_pie = []
                        for class_name in class_counts.keys():
                            # Cập nhật màu sắc cho các lớp mới
                            if class_name in class_names:
                                class_idx = class_names.index(class_name)
                                colors_pie.append(color_hex_mapping[class_idx])
                            else:
                                colors_pie.append('#808080')  # Màu xám cho các loại không xác định

                        wedges, texts, autotexts = ax_pie.pie(
                            list(class_counts.values()),
                            labels=list(class_counts.keys()),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors_pie
                        )
                        # Làm cho văn bản dễ đọc hơn
                        for text in texts:
                            text.set_fontsize(12)
                        for autotext in autotexts:
                            autotext.set_fontsize(12)
                            autotext.set_fontweight('bold')

                        ax_pie.set_title('Tỷ Lệ Các Loại Lúa', fontsize=14)
                        ax_pie.axis('equal')  # Đảm bảo biểu đồ tròn là hình tròn
                        st.pyplot(fig_pie)

                        # Hiển thị thông tin độ tin cậy
                        confidences = results[0].boxes.conf.cpu().numpy()
                        st.markdown("### Phân Phối Độ Tin Cậy")
                        st.write(f"Độ Tin Cậy Trung Bình: {confidences.mean():.2f}")
                        st.write(f"Độ Tin Cậy Thấp Nhất: {confidences.min():.2f}")
                        st.write(f"Độ Tin Cậy Cao Nhất: {confidences.max():.2f}")

                        # Biểu đồ phân phối độ tin cậy
                        fig_conf = plot_confidence_histogram(confidences)
                        st.pyplot(fig_conf)
                    else:
                        st.info("Không phát hiện hạt lúa nào trong hình ảnh.")

                    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Hiển thị hình ảnh mẫu và hướng dẫn khi không có hình ảnh nào được tải lên
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.info("Vui lòng tải lên hình ảnh để bắt đầu quá trình phân loại lúa.")

    # Phần hình ảnh mẫu
    st.markdown("### Hình Ảnh Mẫu")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/placeholder-ob7miW3mUreePYfXdVwkpFWHthzoR5.svg?height=200&width=200",
            caption="Hình Ảnh Lúa Mẫu 1")
    with col2:
        st.image(
            "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/placeholder-ob7miW3mUreePYfXdVwkpFWHthzoR5.svg?height=200&width=200",
            caption="Hình Ảnh Lúa Mẫu 2")
    with col3:
        st.image(
            "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/placeholder-ob7miW3mUreePYfXdVwkpFWHthzoR5.svg?height=200&width=200",
            caption="Hình Ảnh Lúa Mẫu 3")

    st.markdown("### Hướng Dẫn")
    st.markdown("""
    1. Tải lên hình ảnh chứa hạt lúa
    2. Chọn phương án và mô hình phù hợp ở thanh bên
    3. Nhấp 'Xử Lý Hình Ảnh' để chạy phân loại
    4. Xem kết quả và thống kê
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Chân trang
st.markdown("---")
st.markdown('<div class="footer">Hệ Thống Phân Loại Lúa © 2025</div>', unsafe_allow_html=True)
