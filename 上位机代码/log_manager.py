"""
分级日志管理器 — 控制台 + 文件 + UI 回调三重输出
    from log_manager import LogManager
    log = LogManager()
    log.info("串口已连接", source="serial")
    log.error("模型加载失败", source="ai")
    log.debug("帧解码: 5x500mV", source="serial")   # 仅控制台，不写文件
    log.set_ui_callback(my_log_msg_fn)              # 同步到 GUI
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from paths import app_base_dir


# 日志颜色（控制台）
_COLORS = {
    'DEBUG':   '\033[90m',   # 灰色
    'INFO':    '\033[37m',   # 白色
    'WARNING': '\033[33m',   # 黄色
    'ERROR':   '\033[31m',   # 红色
    'RESET':   '\033[0m',
}
_LEVEL_LABELS = {
    'DEBUG':   '调试',
    'INFO':    '运行',
    'WARNING': '警告',
    'ERROR':   '错误',
}


class _UiLogHandler(logging.Handler):
    """将日志桥接到 GUI QPlainTextEdit"""

    def __init__(self):
        super().__init__()
        self._callback = None

    def set_callback(self, fn):
        self._callback = fn

    def emit(self, record):
        if self._callback:
            try:
                ts = datetime.now().strftime("%H:%M:%S")
                level_cn = _LEVEL_LABELS.get(record.levelname, record.levelname)
                src = getattr(record, 'source', '') or ''
                prefix = f"[{level_cn}]" if not src else f"[{level_cn}|{src}]"
                msg = f"{ts} {prefix} {record.getMessage()}"
                self._callback(msg)
            except Exception:
                pass


class LogManager:
    """分级日志管理器 — DEBUG(控制台) / INFO / WARNING / ERROR(控制台+文件)

    用法:
        log = LogManager(log_dir="logs", max_days=30)
        log.set_ui_callback(lambda msg: self.log.appendPlainText(msg))
        log.info("系统启动", source="system")
        log.error("串口写入失败", source="serial")
    """

    def __init__(self, log_dir: str = "logs", max_days: int = 30):
        self._log_dir = os.path.join(app_base_dir(), log_dir)
        self._max_days = max_days
        os.makedirs(self._log_dir, exist_ok=True)

        self._logger = logging.getLogger("CO2Monitor")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # 控制台 handler — DEBUG 及以上（windowed 打包时 sys.stdout 为 None，跳过）
        if sys.stdout is not None:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.DEBUG)
            console.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(console)

        # 文件 handler — INFO 及以上，按日期命名
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self._log_dir, f"co2_monitor_{today}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        ))
        self._logger.addHandler(file_handler)

        # UI handler — INFO 及以上
        self._ui_handler = _UiLogHandler()
        self._ui_handler.setLevel(logging.INFO)
        self._logger.addHandler(self._ui_handler)

        # 启动清理
        self.cleanup_old_logs()

    def set_ui_callback(self, callback):
        """注册 GUI 回调：每条 INFO+ 日志同步到 QPlainTextEdit"""
        self._ui_handler.set_callback(callback)

    def debug(self, msg: str, source: str = ""):
        self._logger.debug(msg, extra={'source': source})

    def info(self, msg: str, source: str = ""):
        self._logger.info(msg, extra={'source': source})

    def warn(self, msg: str, source: str = ""):
        self._logger.warning(msg, extra={'source': source})

    def error(self, msg: str, source: str = ""):
        self._logger.error(msg, extra={'source': source})

    def get_recent_logs(self, n: int = 200) -> list[str]:
        """读取当前日志文件的最近 N 行"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self._log_dir, f"co2_monitor_{today}.log")
        lines = []
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
        except Exception:
            pass
        return [l.rstrip('\n') for l in lines[-n:]]

    def cleanup_old_logs(self, max_days: int = None):
        """清理超过 max_days 天的日志文件"""
        if max_days is None:
            max_days = self._max_days
        import time
        cutoff = time.time() - max_days * 86400
        try:
            for fname in os.listdir(self._log_dir):
                if fname.startswith("co2_monitor_") and fname.endswith(".log"):
                    fpath = os.path.join(self._log_dir, fname)
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
        except Exception:
            pass
