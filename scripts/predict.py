"""
YOLOv8-s 预测脚本
================
命令行批量检测路面图片并生成养护方案报告。
"""
import argparse
import json
import os
from pathlib import Path
from src.utils import load_config
from src.detector import RoadDiseaseDetector
from src.maintenance import MaintenanceDecisionEngine
from src.cold_region import ColdRegionInfo


def predict(
    image_path,
    output_dir="runs/predict",
    temperature=-15,
    road_section="未知道路",
    save_report=True,
):
    """
    检测路面病害并生成养护方案

    Args:
        image_path: 图片路径或目录
        output_dir: 输出目录
        temperature: 温度
        road_section: 路段名称
        save_report: 是否保存报告
    """
    config = load_config()
    detector = RoadDiseaseDetector(config)
    decision_engine = MaintenanceDecisionEngine(config)

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(image_path):
        valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if Path(f).suffix.lower() in valid_ext
        ]
    else:
        image_files = [image_path]

    all_reports = []
    for img_file in image_files:
        print(f"\n[预测] 正在检测: {Path(img_file).name}")

        result = detector.detect_image(img_file, save_result=True, output_dir=output_dir)
        if "error" in result:
            print(f"[预测] 错误: {result['error']}")
            continue

        env = ColdRegionInfo(temperature=temperature)
        plan = decision_engine.generate_plan(
            result["detections"], road_section=road_section, env=env
        )

        report = {
            "image": Path(img_file).name,
            "summary": result["summary"],
            "detections": result["detections"],
            "maintenance_plan": {
                "plan_id": plan.plan_id,
                "road_section": plan.road_section,
                "total_tasks": plan.total_tasks,
                "total_cost": plan.total_cost,
                "total_duration": plan.total_duration,
                "priority_tasks": plan.priority_tasks,
                "overall_risk": plan.overall_risk,
                "recommendations": plan.recommendations,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "disease_name": t.disease_name,
                        "severity": t.severity,
                        "method": t.method,
                        "priority": t.priority,
                        "estimated_cost": t.estimated_cost,
                        "estimated_duration": t.estimated_duration,
                        "cold_region_note": t.cold_region_note,
                    }
                    for t in plan.tasks
                ],
            },
        }
        all_reports.append(report)

        print(f"  病害数量: {result['num_detections']}")
        print(f"  严重程度: {result['summary']['max_severity']}")
        print(f"  养护任务: {plan.total_tasks}")
        print(f"  预估成本: ¥{plan.total_cost:.2f}")

    if save_report and all_reports:
        report_path = os.path.join(output_dir, "maintenance_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        print(f"\n[预测] 报告已保存: {report_path}")

    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-s 路面病害预测")
    parser.add_argument("--input", type=str, required=True, help="图片路径或目录")
    parser.add_argument("--output", type=str, default="runs/predict", help="输出目录")
    parser.add_argument("--temperature", type=float, default=-15, help="当前温度")
    parser.add_argument("--road", type=str, default="未知道路", help="路段名称")

    args = parser.parse_args()
    predict(
        image_path=args.input,
        output_dir=args.output,
        temperature=args.temperature,
        road_section=args.road,
    )
