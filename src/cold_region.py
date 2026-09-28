"""
高寒地区路面病害分析模块
=========================
针对高寒地区特殊气候条件（冻融循环、低温、冰雪等）
对路面病害的影响进行分析和评估。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class ColdRegionInfo:
    """高寒地区环境信息"""
    temperature: float = -15.0       # 当前温度 ℃
    humidity: float = 65.0           # 湿度 %
    freeze_thaw_cycles: int = 0      # 冻融循环次数
    ice_thickness: float = 0.0       # 冰层厚度 mm
    road_age: int = 5                # 道路使用年限
    avg_annual_freeze_days: int = 150  # 年均冻结天数
    permafrost_depth: float = 0.0    # 冻土深度 m
    salt_usage: float = 0.0          # 除冰盐用量 kg/m²


@dataclass
class ColdRegionRisk:
    """高寒地区风险评估结果"""
    freeze_thaw_risk: str = "低"     # 冻融风险
    frost_damage_risk: str = "低"   # 冻胀风险
    ice_snow_risk: str = "低"       # 冰雪风险
    salt_erosion_risk: str = "低"   # 盐冻风险
    overall_risk: str = "低"        # 综合风险
    risk_score: int = 0             # 综合风险评分 0-100
    factors: List[str] = field(default_factory=list)  # 风险因素列表


class ColdRegionAnalyzer:
    """高寒地区路面病害分析器"""

    # 冻融循环影响权重
    FREEZE_THAW_WEIGHTS = {
        "longitudinal_crack": 1.5,
        "transverse_crack": 1.3,
        "block_crack": 1.8,
        "pothole": 2.0,
        "raveling": 1.6,
        "settlement": 1.4,
    }

    # 温度影响因子
    TEMPERATURE_FACTORS = {
        (-999, -30): 2.0,    # 极寒
        (-30, -20): 1.5,     # 严寒
        (-20, -10): 1.2,     # 寒冷
        (-10, -5): 1.0,      # 低温
        (-5, 0): 0.8,        # 冰冻临界
        (0, 999): 0.5,       # 正常
    }

    # 冻融循环阈值
    FREEZE_THAW_THRESHOLDS = {
        "low": 50,
        "medium": 100,
        "high": 150,
        "severe": 200,
    }

    def __init__(self, config=None):
        self.config = config or {}

    def analyze_environment(self, env: ColdRegionInfo) -> ColdRegionRisk:
        """
        分析高寒地区环境风险

        Args:
            env: 高寒地区环境信息

        Returns:
            ColdRegionRisk: 风险评估结果
        """
        risk = ColdRegionRisk()
        score = 0
        factors = []

        # 1. 冻融循环风险评估
        risk.freeze_thaw_risk = self._assess_freeze_thaw(env.freeze_thaw_cycles)
        ft_score = min(env.freeze_thaw_cycles / 200 * 30, 30)
        score += ft_score
        if risk.freeze_thaw_risk != "低":
            factors.append(f"冻融循环{env.freeze_thaw_cycles}次，风险{risk.freeze_thaw_risk}")

        # 2. 冻胀风险评估
        risk.frost_damage_risk = self._assess_frost_damage(env)
        fd_score = min(env.permafrost_depth * 10 + env.road_age * 2, 25)
        score += fd_score
        if risk.frost_damage_risk != "低":
            factors.append(f"冻土深度{env.permafrost_depth}m，道路年限{env.road_age}年")

        # 3. 冰雪风险评估
        risk.ice_snow_risk = self._assess_ice_snow(env)
        is_score = min(env.ice_thickness * 2 + (env.avg_annual_freeze_days / 150) * 15, 20)
        score += is_score
        if risk.ice_snow_risk != "低":
            factors.append(f"冰层厚度{env.ice_thickness}mm，年冻结{env.avg_annual_freeze_days}天")

        # 4. 盐冻风险评估
        risk.salt_erosion_risk = self._assess_salt_erosion(env)
        se_score = min(env.humidity / 100 * 10, 10) if env.humidity > 60 else 0
        score += se_score
        if risk.salt_erosion_risk != "低":
            factors.append(f"湿度{env.humidity}%，除冰盐{env.humidity}kg/m²")

        # 综合评分
        risk.risk_score = int(score)
        risk.factors = factors

        if score >= 70:
            risk.overall_risk = "极高"
        elif score >= 50:
            risk.overall_risk = "高"
        elif score >= 30:
            risk.overall_risk = "中"
        elif score >= 10:
            risk.overall_risk = "低"
        else:
            risk.overall_risk = "极低"

        return risk

    def _assess_freeze_thaw(self, cycles: int) -> str:
        """评估冻融循环风险"""
        if cycles >= self.FREEZE_THAW_THRESHOLDS["severe"]:
            return "极高"
        elif cycles >= self.FREEZE_THAW_THRESHOLDS["high"]:
            return "高"
        elif cycles >= self.FREEZE_THAW_THRESHOLDS["medium"]:
            return "中"
        elif cycles >= self.FREEZE_THAW_THRESHOLDS["low"]:
            return "低"
        return "极低"

    def _assess_frost_damage(self, env: ColdRegionInfo) -> str:
        """评估冻胀风险"""
        risk_value = env.permafrost_depth * 2 + env.road_age * 0.3
        if risk_value >= 10:
            return "极高"
        elif risk_value >= 6:
            return "高"
        elif risk_value >= 3:
            return "中"
        elif risk_value >= 1:
            return "低"
        return "极低"

    def _assess_ice_snow(self, env: ColdRegionInfo) -> str:
        """评估冰雪风险"""
        risk_value = env.ice_thickness * 0.5 + env.avg_annual_freeze_days / 30
        if risk_value >= 8:
            return "极高"
        elif risk_value >= 5:
            return "高"
        elif risk_value >= 3:
            return "中"
        elif risk_value >= 1:
            return "低"
        return "极低"

    def _assess_salt_erosion(self, env: ColdRegionInfo) -> str:
        """评估盐冻剥蚀风险"""
        risk_value = env.humidity / 100 * 2 + env.salt_usage
        if risk_value >= 5:
            return "极高"
        elif risk_value >= 3:
            return "高"
        elif risk_value >= 1.5:
            return "中"
        elif risk_value >= 0.5:
            return "低"
        return "极低"

    def get_temperature_factor(self, temperature: float) -> float:
        """
        获取温度影响因子

        Args:
            temperature: 当前温度 ℃

        Returns:
            float: 温度影响因子
        """
        for (low, high), factor in self.TEMPERATURE_FACTORS.items():
            if low <= temperature < high:
                return factor
        return 1.0

    def adjust_severity(self, base_severity: str, env: ColdRegionInfo) -> str:
        """
        根据高寒地区环境调整病害严重程度

        Args:
            base_severity: 基础严重程度
            env: 环境信息

        Returns:
            str: 调整后的严重程度
        """
        temp_factor = self.get_temperature_factor(env.temperature)
        ft_risk = self.analyze_environment(env)

        severity_order = ["轻微", "中等", "严重", "极严重"]
        base_idx = severity_order.index(base_severity) if base_severity in severity_order else 0

        # 冻融循环加速病害发展
        if ft_risk.risk_score > 50:
            base_idx = min(base_idx + 1, len(severity_order) - 1)
        if temp_factor > 1.5 and base_idx > 0:
            base_idx = min(base_idx + 1, len(severity_order) - 1)

        return severity_order[base_idx]

    def get_maintenance_window(self, env: ColdRegionInfo) -> dict:
        """
        计算最佳养护窗口期

        Args:
            env: 环境信息

        Returns:
            dict: 养护窗口信息
        """
        if env.temperature < self.config.get("maintenance", {}).get(
            "temperature", {}
        ).get("construction_min", 5):
            return {
                "suitable": False,
                "reason": "当前温度低于施工最低温度，不建议进行养护施工",
                "recommended_season": "建议在 5月-9月 气温高于 5℃ 时施工",
                "urgency": "如有紧急病害，建议采用冷补材料临时处理",
            }
        else:
            return {
                "suitable": True,
                "reason": "当前气温适合进行养护施工",
                "recommended_methods": ["热拌沥青", "微表处", "灌缝"],
                "urgency": "正常施工",
            }
