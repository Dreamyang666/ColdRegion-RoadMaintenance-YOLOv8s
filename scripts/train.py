"""
YOLOv8-s 模型训练脚本
=====================
用于在路面病害数据集上微调 YOLOv8-s 模型。
"""
import argparse
from ultralytics import YOLO


def train(
    data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    device="0",
    weights="yolov8s.pt",
    project="runs/train",
    name="road_disease",
):
    """
    训练 YOLOv8-s 模型

    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 输入图片尺寸
        batch: 批次大小
        device: 设备 (cpu/0/0,1)
        weights: 预训练权重
        project: 结果保存目录
        name: 实验名称
    """
    print("=" * 60)
    print("  YOLOv8-s 路面病害检测模型训练")
    print("=" * 60)
    print(f"  数据集: {data_yaml}")
    print(f"  预训练权重: {weights}")
    print(f"  训练轮数: {epochs}")
    print(f"  图片尺寸: {imgsz}")
    print(f"  批次大小: {batch}")
    print(f"  设备: {device}")
    print("=" * 60)

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        augment=True,
        # 高寒地区数据增强
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n训练完成！")
    print(f"模型已保存至: {project}/{name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-s 路面病害检测训练")
    parser.add_argument("--data", type=str, default="data/datasets/road_disease.yaml", help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str, default="0", help="设备 cpu/0/0,1")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="预训练权重")

    args = parser.parse_args()
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        weights=args.weights,
    )
