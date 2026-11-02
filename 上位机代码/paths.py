"""
路径辅助模块 — 兼容 PyInstaller 打包环境

- app_base_dir(): 可写数据目录（DB/配置/日志）
    打包后 → exe 所在目录（数据持久保存在 exe 旁）
    开发时 → 模块所在目录
- resource_dir(): 只读资源目录（内置模型等）
    打包后 → _MEIPASS 临时解压目录
    开发时 → 模块所在目录
"""
import os
import sys


def app_base_dir() -> str:
    """返回应用数据目录：打包后为 exe 所在目录，开发时为模块所在目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_dir() -> str:
    """返回只读资源目录：打包后为 _MEIPASS，开发时为模块所在目录"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))
