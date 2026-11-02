"""
上位机统一启动入口
    python run.py
等价于 python co2_monitor.py，但提供版本信息和依赖检查
"""
import sys
import os

VERSION = "1.0.0"
APP_NAME = "CO2 呼吸波形监控上位机"


def check_dependencies():
    """检查核心依赖是否可导入，缺失则提示"""
    required = ["PySide6", "pyqtgraph", "onnxruntime", "numpy"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[WARN] 缺少依赖: {', '.join(missing)}")
        print("      请运行: pip install -r requirements.txt")
        return False
    return True


def main():
    if sys.platform == "win32":
        # 打包为 windowed 模式时 sys.stdout/stderr 为 None，需判空
        try:
            if sys.stdout is not None:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if sys.stderr is not None:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"══════════════════════════════════════════")
    print(f"  {APP_NAME}  v{VERSION}")
    print(f"══════════════════════════════════════════")

    if not check_dependencies():
        # 依赖缺失仍尝试启动（PySide6 必须存在才能继续）
        pass

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)

    from co2_monitor import CO2MonitorWindow
    win = CO2MonitorWindow()
    win.show()
    win.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 确保工作目录正确
    main()
