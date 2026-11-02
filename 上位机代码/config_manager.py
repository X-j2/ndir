"""
JSON 配置文件读写模块 — 保存通信参数、用户设置
    from config_manager import ConfigManager
    cfg = ConfigManager("config.json")
    baud = cfg.get("serial.baud_rate", 115200)
    cfg.set("serial.port", "COM3")
    cfg.save()
"""
import json
import os
import copy
from threading import Lock

from paths import app_base_dir


DEFAULT_CONFIG = {
    "serial": {
        "port": "",
        "baud_rate": 115200,
        "data_bits": 8,
        "stop_bits": 1,
        "parity": "none",
        "flow_control": "none",
    },
    "app": {
        "window_width": 1000,
        "window_height": 750,
        "auto_connect": False,
        "db_retention_days": 30,
    },
    "alarm": {
        "enabled": True,
        "consecutive_threshold": 2,
        "min_confidence": 0.6,
        "sound_enabled": False,
    },
    "ai": {
        "model_path": "etco2_model_v2.onnx",
        "inference_interval_sec": 30,
    },
}


class ConfigManager:
    """JSON 配置文件管理器 — 支持点号路径读写"""

    def __init__(self, config_path: str = "config.json"):
        self._config_path = os.path.join(app_base_dir(), config_path)
        self._lock = Lock()
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    def load(self) -> dict:
        """加载配置文件，损坏时用默认值重建"""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # 递归合并：保留 DEFAULT_CONFIG 中新增的键
                self._merge_defaults(loaded, DEFAULT_CONFIG)
                self._data = loaded
        except (json.JSONDecodeError, IOError, OSError):
            # 文件损坏或不可读 → 重建
            self._data = copy.deepcopy(DEFAULT_CONFIG)
            self.save()
        return dict(self._data)

    def save(self):
        """保存配置文件"""
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self._config_path) or ".", exist_ok=True)
                with open(self._config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
        except (IOError, OSError):
            pass

    def get(self, key_path: str, default=None):
        """按点号路径读取配置值，如 'serial.baud_rate'"""
        keys = key_path.split('.')
        node = self._data
        try:
            for k in keys:
                node = node[k]
            return node
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value):
        """按点号路径设置配置值"""
        keys = key_path.split('.')
        node = self._data
        try:
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value
        except (KeyError, TypeError, IndexError):
            pass

    def reset_to_defaults(self):
        """恢复默认配置"""
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        self.save()

    def _merge_defaults(self, loaded: dict, defaults: dict):
        """递归补全加载配置中缺失的默认键"""
        for key, default_val in defaults.items():
            if key not in loaded:
                loaded[key] = copy.deepcopy(default_val)
            elif isinstance(default_val, dict) and isinstance(loaded[key], dict):
                self._merge_defaults(loaded[key], default_val)
