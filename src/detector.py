"""
YOLOv8-s 路面病害检测核心模块
=============================
基于 Ultralytics YOLOv8-s 模型，针对高寒地区道路病害进行检测。
支持图片、视频和实时摄像头检测。
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.utils import format_detection_result, DISEASE_COLORS


class RoadDiseaseDetector:
    """路面病害检测器"""

    def __init__(self, config=None):
        """
        初始化检测器

        Args:
            config: 配置字典，包含模型路径、置信度阈值等
        """
        self.model_name = config.get("model", {}).get("name", "yolov8s") if config else "yolov8s"
        self.weights = config.get("model", {}).get("weights", "yolov8s.pt") if config else "yolov8s.pt"
        self.imgsz = config.get("model", {}).get("imgsz", 640) if config else 640
        self.conf_threshold = config.get("model", {}).get("conf", 0.25) if config else 0.25
        self.iou_threshold = config.get("model", {}).get("iou", 0.45) if config else 0.45

        # 类别名称映射
        self.class_names = config.get("classes", []) if config else []
        self.cold_region_names = config.get("cold_region_classes", []) if config else []
        self.all_names = self.class_names + self.cold_region_names

        self.names_dict = {i: name for i, name in enumerate(self.all_names)}

        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 YOLOv8-s 模型"""
        print(f"[检测器] 正在加载模型: {self.weights}")
        try:
            self.model = YOLO(self.weights)
            print(f"[检测器] 模型加载成功，类别数: {len(self.all_names)}")
        except Exception as e:
            print(f"[检测器] 模型加载失败: {e}")
            print("[检测器] 尝试使用预训练 yolov8s 模型...")
            self.model = YOLO("yolov8s.pt")
            print("[检测器] 预训练模型加载成功（COCO 80类）")
            self.names_dict = self.model.names

    def detect_image(self, image_path, save_result=True, output_dir="runs/detect"):
        """
        检测单张图片

        Args:
            image_path: 图片路径
            save_result: 是否保存检测结果图片
            output_dir: 结果保存目录

        Returns:
            dict: 检测结果，包含绘制后的图片和病害列表
        """
        if not os.path.exists(image_path):
            return {"error": f"图片路径不存在: {image_path}"}

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"error": f"无法读取图片: {image_path}"}

        image_area = image.shape[0] * image.shape[1]

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        annotated = results[0].plot()
        detections = format_detection_result(results[0].boxes, self.names_dict, image_area)

        output_path = None
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{filename}_detected.jpg")
            cv2.imwrite(output_path, annotated)

        summary = self._generate_summary(detections, image_area)

        return {
            "image_path": image_path,
            "output_path": output_path,
            "image_shape": image.shape,
            "detections": detections,
            "num_detections": len(detections),
            "summary": summary,
            "annotated_image": annotated,
        }

    def detect_video(self, video_path, output_path=None, save_result=True):
        """
        检测视频

        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            save_result: 是否保存结果

        Returns:
            dict: 检测结果统计
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"无法打开视频: {video_path}"}

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if save_result and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image_area = frame.shape[0] * frame.shape[1]

            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )

            annotated = results[0].plot()
            detections = format_detection_result(results[0].boxes, self.names_dict, image_area)

            for d in detections:
                d["frame"] = frame_count
            all_detections.extend(detections)

            if save_result and output_path:
                out.write(annotated)

            if frame_count % 50 == 0:
                print(f"[检测器] 已处理 {frame_count}/{total_frames} 帧")

        cap.release()
        if save_result and output_path:
            out.release()

        return {
            "video_path": video_path,
            "output_path": output_path,
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "total_detections": len(all_detections),
            "detections": all_detections,
        }

    def detect_batch(self, image_dir, output_dir="runs/detect/batch"):
        """
        批量检测目录下所有图片

        Args:
            image_dir: 图片目录
            output_dir: 输出目录

        Returns:
            list: 所有图片的检测结果
        """
        valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in valid_ext
        ]

        if not image_files:
            return {"error": f"目录下未找到图片: {image_dir}"}

        os.makedirs(output_dir, exist_ok=True)
        all_results = []

        for img_file in image_files:
            print(f"[检测器] 正在检测: {Path(img_file).name}")
            result = self.detect_image(img_file, save_result=True, output_dir=output_dir)
            all_results.append(result)

        total_detections = sum(r.get("num_detections", 0) for r in all_results)
        print(f"[检测器] 批量检测完成: {len(image_files)} 张图片, 共 {total_detections} 个病害")

        return {
            "total_images": len(image_files),
            "total_detections": total_detections,
            "results": all_results,
        }

    def _generate_summary(self, detections, image_area):
        """生成检测摘要"""
        if not detections:
            return {
                "total": 0,
                "diseases": {},
                "max_severity": "无病害",
                "coverage": 0.0,
            }

        disease_count = {}
        total_area_ratio = 0

        for det in detections:
            name = det["disease_name"]
            disease_count[name] = disease_count.get(name, 0) + 1
            total_area_ratio += det["area_ratio"]

        return {
            "total": len(detections),
            "diseases": disease_count,
            "max_severity": self._get_max_severity(detections),
            "coverage": round(total_area_ratio * 100, 2),
        }

    @staticmethod
    def _get_max_severity(detections):
        """获取最大病害严重程度"""
        if not detections:
            return "无病害"
        max_conf = max(d["confidence"] for d in detections)
        max_area = max(d["area_ratio"] for d in detections)

        if max_area > 0.30 or max_conf > 0.85:
            return "严重"
        elif max_area > 0.15 or max_conf > 0.60:
            return "中等"
        else:
            return "轻微"
