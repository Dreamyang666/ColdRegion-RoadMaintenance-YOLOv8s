"""
快速启动脚本
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("  高寒地区道路智能养护系统")
    print("  基于 YOLOv8-s")
    print("=" * 60)
    print("  启动中...")
    print("  访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
