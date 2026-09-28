"""
高寒地区道路智能养护系统 - Web 服务
===================================
Flask 应用，提供路面病害检测和养护方案生成的可视化界面。
"""
import os
import sys
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logger
from src.detector import RoadDiseaseDetector
from src.maintenance import MaintenanceDecisionEngine
from src.cold_region import ColdRegionInfo, ColdRegionAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

config = load_config()
logger = setup_logger()

print("[系统] 正在初始化检测器和决策引擎...")
detector = RoadDiseaseDetector(config)
decision_engine = MaintenanceDecisionEngine(config)
cold_analyzer = ColdRegionAnalyzer(config)
print("[系统] 初始化完成！")


@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """图片检测接口"""
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(filename)[1] or ".jpg"
    saved_filename = f"upload_{timestamp}{ext}"
    filepath = os.path.join(UPLOAD_DIR, saved_filename)
    file.save(filepath)

    temperature = float(request.form.get("temperature", -15))
    road_section = request.form.get("road_section", "未知道路")
    freeze_thaw_cycles = int(request.form.get("freeze_thaw_cycles", 80))
    humidity = float(request.form.get("humidity", 65))
    ice_thickness = float(request.form.get("ice_thickness", 0))
    road_age = int(request.form.get("road_age", 5))
    avg_freeze_days = int(request.form.get("avg_freeze_days", 150))
    permafrost_depth = float(request.form.get("permafrost_depth", 0))
    salt_usage = float(request.form.get("salt_usage", 0))

    try:
        result = detector.detect_image(filepath, save_result=True, output_dir=RESULT_DIR)

        if "error" in result:
            return jsonify(result), 500

        env = ColdRegionInfo(
            temperature=temperature,
            humidity=humidity,
            freeze_thaw_cycles=freeze_thaw_cycles,
            ice_thickness=ice_thickness,
            road_age=road_age,
            avg_annual_freeze_days=avg_freeze_days,
            permafrost_depth=permafrost_depth,
            salt_usage=salt_usage,
        )

        plan = decision_engine.generate_plan(
            result["detections"], road_section=road_section, env=env
        )

        risk = cold_analyzer.analyze_environment(env)

        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        cv2.imwrite(result_path, result["annotated_image"])

        tasks_data = []
        for task in plan.tasks:
            tasks_data.append({
                "task_id": task.task_id,
                "disease_name": task.disease_name,
                "severity": task.severity,
                "location": task.location,
                "area": task.area,
                "method": task.method,
                "priority": task.priority,
                "estimated_cost": task.estimated_cost,
                "estimated_duration": task.estimated_duration,
                "cold_region_note": task.cold_region_note,
                "urgent": task.urgent,
            })

        return jsonify({
            "success": True,
            "result_image": f"/static/results/{result_filename}",
            "summary": result["summary"],
            "detections": result["detections"],
            "num_detections": result["num_detections"],
            "maintenance_plan": {
                "plan_id": plan.plan_id,
                "road_section": plan.road_section,
                "generate_time": plan.generate_time,
                "total_tasks": plan.total_tasks,
                "total_cost": plan.total_cost,
                "total_duration": plan.total_duration,
                "priority_tasks": plan.priority_tasks,
                "tasks": tasks_data,
                "overall_risk": plan.overall_risk,
                "recommendations": plan.recommendations,
            },
            "risk_assessment": {
                "freeze_thaw_risk": risk.freeze_thaw_risk,
                "frost_damage_risk": risk.frost_damage_risk,
                "ice_snow_risk": risk.ice_snow_risk,
                "salt_erosion_risk": risk.salt_erosion_risk,
                "overall_risk": risk.overall_risk,
                "risk_score": risk.risk_score,
                "risk_factors": risk.factors,
            },
        })

    except Exception as e:
        logger.error(f"检测失败: {e}")
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@app.route("/batch", methods=["POST"])
def batch_detect():
    """批量检测接口"""
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "未上传文件"}), 400

    results = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        result = detector.detect_image(filepath, save_result=True, output_dir=RESULT_DIR)
        results.append({
            "filename": filename,
            "summary": result.get("summary", {}),
            "num_detections": result.get("num_detections", 0),
        })

    return jsonify({"success": True, "results": results})


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({"status": "ok", "model": config.get("model", {}).get("name", "yolov8s")})


if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 5000)
    debug = config.get("server", {}).get("debug", True)
    app.run(host=host, port=port, debug=debug)
