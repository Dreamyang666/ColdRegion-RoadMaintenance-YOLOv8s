"""工具函数模块"""
import os
import yaml
import logging
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir="logs"):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# 病害中英文映射
DISEASE_MAPPING = {
    "longitudinal_crack": "纵向裂缝",
    "transverse_crack": "横向裂缝",
    "block_crack": "块状裂缝",
    "pothole": "坑槽",
    "raveling": "松散",
    "settlement": "沉陷",
    "bleeding": "泛油",
    "patch": "修补",
    "freeze_thaw_crack": "冻融裂缝",
    "frost_heave": "冰冻隆起",
    "frost_pothole": "冻胀坑槽",
    "ice_cover": "冰雪覆盖",
    "salt_erosion": "盐冻剥蚀",
}

# 病害颜色 (BGR格式)
DISEASE_COLORS = {
    "纵向裂缝": (0, 0, 255),
    "横向裂缝": (0, 165, 255),
    "块状裂缝": (0, 255, 255),
    "坑槽": (0, 0, 128),
    "松散": (255, 0, 255),
    "沉陷": (255, 165, 0),
    "泛油": (128, 0, 128),
    "修补": (0, 128, 0),
    "冻融裂缝": (0, 0, 200),
    "冰冻隆起": (200, 0, 0),
    "冻胀坑槽": (0, 100, 200),
    "冰雪覆盖": (255, 255, 255),
    "盐冻剥蚀": (100, 100, 100),
}


def calculate_iou(box1, box2):
    """计算两个边界框的交并比 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def format_detection_result(boxes, names, image_area):
    """格式化检测结果"""
    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / image_area if image_area > 0 else 0

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        disease_name = names.get(cls_id, f"未知_{cls_id}")

        results.append({
            "id": i,
            "class_id": cls_id,
            "disease_name": disease_name,
            "confidence": round(conf, 3),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "area_ratio": round(area_ratio, 4),
        })
    return results
