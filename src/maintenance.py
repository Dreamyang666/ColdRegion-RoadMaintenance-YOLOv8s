"""
路面智能养护决策模块
====================
根据 YOLOv8-s 检测结果和高寒地区环境分析，
自动生成养护方案、优先级排序和成本估算。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from src.cold_region import ColdRegionInfo, ColdRegionAnalyzer


@dataclass
class MaintenanceTask:
    """养护任务"""
    task_id: str
    disease_name: str           # 病害名称
    severity: str              # 严重程度: 轻微/中等/严重/极严重
    location: str              # 位置描述
    area: float                # 面积 m²
    method: str                # 养护方法
    priority: int              # 优先级 1-5 (5最高)
    estimated_cost: float      # 预估成本 元
    estimated_duration: int     # 预估工期 天
    cold_region_note: str = "" # 高寒地区特别注意事项
    urgent: bool = False       # 是否紧急


@dataclass
class MaintenancePlan:
    """养护方案"""
    plan_id: str
    road_section: str          # 路段
    generate_time: str        # 生成时间
    total_tasks: int          # 总任务数
    total_cost: float         # 总成本
    total_duration: int       # 总工期
    priority_tasks: int       # 高优先级任务数
    tasks: List[MaintenanceTask] = field(default_factory=list)
    overall_risk: str = "低"
    recommendations: List[str] = field(default_factory=list)


class MaintenanceDecisionEngine:
    """养护决策引擎"""

    # 病害养护方法库
    MAINTENANCE_METHODS = {
        "纵向裂缝": {
            "轻微": ("灌缝", 35, 0.5),
            "中等": ("灌缝+贴缝带", 80, 1),
            "严重": ("开槽灌缝+抗裂贴", 150, 2),
            "极严重": ("铣刨重铺", 350, 5),
        },
        "横向裂缝": {
            "轻微": ("灌缝", 30, 0.5),
            "中等": ("灌缝+贴缝带", 70, 1),
            "严重": ("开槽灌缝+抗裂贴", 140, 2),
            "极严重": ("铣刨重铺", 300, 5),
        },
        "块状裂缝": {
            "轻微": ("表面封层", 45, 1),
            "中等": ("微表处", 120, 2),
            "严重": ("铣刨+面层重铺", 280, 4),
            "极严重": ("整段铣刨重铺", 500, 7),
        },
        "坑槽": {
            "轻微": ("冷补料填补", 50, 0.5),
            "中等": ("热补料修补", 120, 1),
            "严重": ("深坑修补+基层处理", 250, 3),
            "极严重": ("全深度修补", 400, 5),
        },
        "松散": {
            "轻微": ("表面封层", 40, 1),
            "中等": ("微表处", 100, 2),
            "严重": ("铣刨+薄层罩面", 220, 3),
            "极严重": ("铣刨+面层重铺", 400, 5),
        },
        "沉陷": {
            "轻微": ("注浆加固", 80, 2),
            "中等": ("基层换填+面层修补", 200, 4),
            "严重": ("路基处理+面层重铺", 350, 7),
            "极严重": ("整体翻修", 600, 15),
        },
        "泛油": {
            "轻微": ("撒布石屑", 20, 0.5),
            "中等": ("微表处", 80, 1),
            "严重": ("铣刨+薄层罩面", 200, 3),
            "极严重": ("铣刨重铺", 350, 5),
        },
        "冻融裂缝": {
            "轻微": ("低温灌缝", 40, 1),
            "中等": ("低温灌缝+抗裂贴", 100, 1),
            "严重": ("开槽灌缝+防水层", 180, 3),
            "极严重": ("铣刨重铺+防水层", 400, 6),
        },
        "冰冻隆起": {
            "轻微": ("注浆抬升", 100, 2),
            "中等": ("基层换填+面层修补", 250, 4),
            "严重": ("路基排水+面层重铺", 400, 7),
            "极严重": ("整体翻修+排水系统", 700, 15),
        },
        "冻胀坑槽": {
            "轻微": ("冷补料+防冻层", 60, 1),
            "中等": ("热补+防冻层修补", 150, 2),
            "严重": ("深坑修补+防冻层", 300, 4),
            "极严重": ("全深度修补+排水", 500, 7),
        },
        "盐冻剥蚀": {
            "轻微": ("表面修复剂", 35, 1),
            "中等": ("微表处+防腐剂", 110, 2),
            "严重": ("铣刨+防腐面层", 250, 4),
            "极严重": ("铣刨重铺+防腐处理", 450, 6),
        },
    }

    # 优先级判定阈值
    PRIORITY_THRESHOLDS = {
        "极严重": 5,
        "严重": 4,
        "中等": 3,
        "轻微": 2,
    }

    # 高寒地区特别注意事项
    COLD_REGION_NOTES = {
        "纵向裂缝": "高寒地区温差大，灌缝材料须选用低温柔韧性好的改性沥青",
        "横向裂缝": "冻融循环加速裂缝扩展，建议增加抗裂贴加强",
        "块状裂缝": "冻融作用下块状裂缝发展快，建议雨季前完成养护",
        "坑槽": "冬季坑槽修补须使用冷补料，严禁低温热补施工",
        "松散": "盐冻和冻融加速松散，建议使用高性能改性乳化沥青",
        "沉陷": "冻胀融沉交替导致沉陷加重，需同步处理排水系统",
        "冻融裂缝": "须使用耐低温灌缝材料，推荐-30℃仍保持柔性的材料",
        "冰冻隆起": "须先解决路基排水和防冻层问题，再处理面层",
        "冻胀坑槽": "修补时需恢复防冻层，防止再次冻胀",
        "盐冻剥蚀": "须彻底清除盐蚀层，使用耐盐腐蚀材料修补",
    }

    def __init__(self, config=None):
        self.config = config or {}
        self.cold_region_analyzer = ColdRegionAnalyzer(config)

    def generate_plan(
        self,
        detections: List[Dict],
        road_section: str = "未知道路",
        env: Optional[ColdRegionInfo] = None,
    ) -> MaintenancePlan:
        """
        根据检测结果生成养护方案

        Args:
            detections: YOLOv8-s 检测结果列表
            road_section: 路段名称
            env: 高寒地区环境信息

        Returns:
            MaintenancePlan: 养护方案
        """
        if env is None:
            env = ColdRegionInfo()

        cold_risk = self.cold_region_analyzer.analyze_environment(env)
        tasks = []
        task_counter = 0

        for det in detections:
            disease_name = det.get("disease_name", "未知病害")
            confidence = det.get("confidence", 0)
            area_ratio = det.get("area_ratio", 0)

            severity = self._determine_severity(disease_name, confidence, area_ratio, env)

            if env.temperature < 0:
                severity = self.cold_region_analyzer.adjust_severity(severity, env)

            method_info = self.MAINTENANCE_METHODS.get(disease_name, {}).get(
                severity, ("观察监测", 10, 1)
            )
            method_name, unit_cost, duration = method_info

            estimated_area = max(area_ratio * 10, 0.5)  # 估算面积 m²
            estimated_cost = round(estimated_area * unit_cost, 2)

            priority = self.PRIORITY_THRESHOLDS.get(severity, 1)
            urgent = priority >= 4

            note = self.COLD_REGION_NOTES.get(disease_name, "")

            task_counter += 1
            bbox = det.get("bbox", [0, 0, 0, 0])
            location = f"区域({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"

            task = MaintenanceTask(
                task_id=f"T{task_counter:03d}",
                disease_name=disease_name,
                severity=severity,
                location=location,
                area=round(estimated_area, 2),
                method=method_name,
                priority=priority,
                estimated_cost=estimated_cost,
                estimated_duration=duration,
                cold_region_note=note,
                urgent=urgent,
            )
            tasks.append(task)

        tasks.sort(key=lambda t: t.priority, reverse=True)

        total_cost = sum(t.estimated_cost for t in tasks)
        total_duration = sum(t.estimated_duration for t in tasks)
        priority_count = sum(1 for t in tasks if t.priority >= 4)

        recommendations = self._generate_recommendations(tasks, cold_risk, env)

        return MaintenancePlan(
            plan_id=f"MP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            road_section=road_section,
            generate_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tasks=len(tasks),
            total_cost=round(total_cost, 2),
            total_duration=total_duration,
            priority_tasks=priority_count,
            tasks=tasks,
            overall_risk=cold_risk.overall_risk,
            recommendations=recommendations,
        )

    def _determine_severity(self, disease_name, confidence, area_ratio, env):
        """确定病害严重程度"""
        # 面积比阈值
        severity_cfg = self.config.get("maintenance", {}).get("severity_levels", {})
        minor_threshold = severity_cfg.get("minor", {}).get("area_ratio", 0.05)
        moderate_threshold = severity_cfg.get("moderate", {}).get("area_ratio", 0.15)
        severe_threshold = severity_cfg.get("severe", {}).get("area_ratio", 0.30)

        if area_ratio >= severe_threshold or confidence > 0.85:
            severity = "极严重" if area_ratio >= 0.50 else "严重"
        elif area_ratio >= moderate_threshold or confidence > 0.60:
            severity = "严重"
        elif area_ratio >= minor_threshold or confidence > 0.40:
            severity = "中等"
        else:
            severity = "轻微"

        return severity

    def _generate_recommendations(self, tasks, cold_risk, env):
        """生成养护建议"""
        recommendations = []

        # 高优先级任务建议
        urgent_tasks = [t for t in tasks if t.urgent]
        if urgent_tasks:
            recommendations.append(
                f"发现 {len(urgent_tasks)} 个紧急病害，建议 7 天内完成处理"
            )

        # 冻融风险建议
        if cold_risk.risk_score >= 50:
            recommendations.append(
                "高寒地区冻融风险较高，建议在冻融期前（9-10月）完成预防性养护"
            )

        # 温度建议
        temp_window = self.cold_region_analyzer.get_maintenance_window(env)
        if not temp_window["suitable"]:
            recommendations.append(temp_window["reason"])
            recommendations.append(temp_window["recommended_season"])
        else:
            recommendations.append("当前气温适合养护施工，可正常开展作业")

        # 预防性养护建议
        if len(tasks) > 5:
            recommendations.append("病害密度较高，建议增加预防性养护（雾封层/微表处）")

        # 排水系统建议
        if cold_risk.frost_damage_risk in ("高", "极高"):
            recommendations.append("冻胀风险高，建议检查并完善路基排水系统")

        if not recommendations:
            recommendations.append("路况良好，建议保持定期巡检")

        return recommendations
