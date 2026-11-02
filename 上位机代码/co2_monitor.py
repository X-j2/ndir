"""
CO2 呼吸波形监控上位机
- 模拟模式：生成仿真的 CO2 波形 + 用 ONNX 模型做真正的 AI 推理
- 硬件模式：接收 STM32 串口数据并显示

依赖：pip install PySide6 pyqtgraph pyserial onnxruntime numpy
运行：python co2_monitor.py
"""
import sys
import random
import os
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QPlainTextEdit, QFrame, QSplitter,
    QSystemTrayIcon, QMenu, QTableWidget, QTableWidgetItem,
    QDateEdit, QHeaderView, QTabWidget, QSlider,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, Slot
from PySide6.QtGui import QFont, QIcon

import pyqtgraph as pg
from PySide6.QtSerialPort import QSerialPort, QSerialPortInfo

# ── ONNX Runtime ──
import onnxruntime as ort

# ── 寻峰（与训练端 v2 extract_features 一致）──
from scipy.signal import find_peaks

# ── 数据库 / 配置 / 日志模块 ──
from db_manager import DatabaseManager
from config_manager import ConfigManager
from log_manager import LogManager
from report_generator import ReportGenerator
from excel_exporter import ExcelExporter
from paths import app_base_dir, resource_dir
from host_core.runtime import Stage3Runtime
from host_core.replay import ReplayPack, ReplaySession
from host_core.stage4_runtime import Stage4Runtime

# ── Stage 3 相对模式 / 事件引擎 ──
DISPLAY_MODE_RELATIVE = "relative"
EVENT_LABEL = "呼气 CO2 低值/消失样事件"

# ── AI 标签映射（与 v2.2 模型一致：4 类，标签 0/1/2/3 = 正常/急促/屏息/呼吸异常）
LABEL_MAP = {0: "正常呼吸", 1: "急促呼吸", 2: "屏息", 3: "呼吸异常"}

LABEL_COLORS = {
    0: "#00CC66", 1: "#FF9933", 2: "#FF4444", 3: "#FFDD44",
}

# ── 模拟呼吸模式（"真实标签"，用于对比模型预测）──
# 参数为真实呼吸波形（capnogram）语义，经离线标定：用同一 ONNX 模型验证，
# 每个模式 16 个随机种子下模型预测均命中 truth 标签（16/16），保证模拟数据的
# 诊断结论与模式一致。
#   rr     : 呼吸率（次/分）
#   etco2  : 呼气末 CO2（平台高度，mmHg）
#   base   : 吸气相基线（mmHg，接近 0）
#   f3     : 呼气平台占呼吸周期的比例
#   duration: 单模式持续秒数 —— 切换时会清空推理窗口，故前 90 秒（AI 窗口）
#             处于"预热中"，只有后段给出本模式的诊断。设为 180s 可保证每个模式
#             有约 90 秒的有效诊断展示时间。
SIM_PATTERNS = [
    {"name": "正常呼吸", "truth": 0, "rr": 16.0, "etco2": 36.0, "base": 6.0, "f3": 0.55, "duration": 180},
    {"name": "急促呼吸", "truth": 1, "rr": 28.0, "etco2": 35.0, "base": 1.0, "f3": 0.60, "duration": 180},
    {"name": "屏息",     "truth": 2, "rr": 0.5,  "etco2": 3.0,  "base": 0.5, "f3": 0.60, "duration": 180},
    {"name": "呼吸异常", "truth": 3, "rr": 12.0, "etco2": 30.0, "base": 1.0, "f3": 0.60, "duration": 180},
]

# ── ONNX 模型路径（打包后从资源目录读取）──
MODEL_PATH = os.path.join(resource_dir(), "etco2_model_v2.onnx")

# ── v2 模型 AI 常量（12 维特征 / 4 类 / 90s 窗口，与训练端一致）──
AI_SAMPLE_RATE = 25.0          # 上位机采样率 Hz（5点×200ms）
AI_WINDOW_SECONDS = 90.0       # 推理窗口长度（训练用 90s）
AI_WINDOW_POINTS = int(AI_WINDOW_SECONDS * AI_SAMPLE_RATE)  # 2250 点
AI_MIN_POINTS = 450            # 少于该点数不推理（20% 窗口）
AI_INFERENCE_INTERVAL_MS = 30000  # 推理定时器周期（30s 滑窗）
# 窗口未满时的"预热"提示：实测窗口 <60s 时诊断不可靠（急促易误判为正常），
# 故未满窗时结果仅作参考，且不触发告警通知。
ALARM_NOTIFY_COOLDOWN_SEC = 60  # 告警弹窗/蜂鸣最小间隔（秒），防止刷屏

# ── 帧协议常量（与 STM32 一致）
# 新布局 (v2.3, 带帧头+校验和+队列溢出计数):
#   byte 0     : FRAME_HEADER = 0xFF
#   byte 1-10  : 5 个 CO₂ 浓度 (每值 2 字节, 大端序, 单位 ppm)
#   byte 11    : AI 有效标志 (0xAA 有 / 0x00 无)
#   byte 12    : AI 分类 (0/1/2/3)
#   byte 13-14 : 呼吸率×10 (2 字节大端)
#   byte 15    : 校验和 (byte0~14 累加取低 8 位)
#   byte 16-19 : 采样队列累计丢弃点数 (4 字节大端, 正常恒为 0)
#   byte 20-63 : 0x00 填充
#
# ⚠ v2.3 变更 (2026-09)：byte 1-10 由「ACT 通道电压(mV)」改为「CO₂ 浓度(ppm)」。
#
# 为什么改：原先本端只拿到 ACT 一路绝对电压，用它做了一套独立标定
# (×VOLTAGE_GAIN×CO2_FACTOR → mmHg)；而固件用的是 act/ref 比值经两点标定得
# ppm。比值法能消除光源老化的影响，绝对电压法不能——两套标定互不相干，
# 本端画出的波形本身就比固件差。改为直接收浓度后，两端同源。
#
# ⚠ 帧长度、字节偏移、校验和均未变，因此校验和照样通过。
#   固件 v2.3 配本端 v2.2 及更早版本时，会把 ppm 当成电压
#   (ppm×0.0015 ≈ 伪 mmHg)，数值完全错误却不报错。**两端必须同步升级。**
#
# byte 16-19 由 F407 固件 v2.2 起新增。固件把采样放在 TIM3 定时器中断里
# 以精确 25Hz 产生、主循环异步消费；若主循环阻塞过久（例如端侧 AI 满窗
# 推理超时）导致队列满，采样点会被丢弃并累加此计数。
# 计数 > 0 表示该时间段的波形存在缺口，据此得出的 AI 结论不可信。
FRAME_SIZE = 64           # 每帧字节数
FRAME_HEADER = 0xFF       # 帧头魔数 (byte 0)
SAMPLES_PER_FRAME = 5     # 每帧采样点数
CONCENTRATION_START = 1   # 浓度区起始字节 (byte 1, 每浓度 2 字节)
AI_FLAG_INDEX = 11        # AI 有效标志位 (byte 11)
AI_CLASS_INDEX = 12       # AI 分类 (byte 12)
RR_INDEX = 13             # 呼吸率起始字节 (byte 13-14, 大端)
CHECKSUM_INDEX = 15       # 校验和 (byte 15)
OVERFLOW_INDEX = 16       # 采样队列累计丢弃点数起始字节 (byte 16-19, 大端)
AI_MAGIC = 0xAA           # AI 结果有效标志 (byte 11)
PPM_TO_MMHG = 0.00076     # ppm → mmHg 比例 (760/1e6, 标准大气压)
                          # 与固件 ai_bridge.c 的 AB_PPM_TO_MMHG_SCALE 保持一致
STALE_FRAME_LIMIT = 3     # 连续无法解析的帧数上限，超过后尝试帧同步恢复
OVERFLOW_ALERT_POINTS = 3 # 累计丢点达到该值时记录告警（3 点 ≈ 0.12 秒）


# ── 协议编解码（纯函数，无 Qt 依赖，可脱离 GUI 独立测试）──

def ppm_to_mmhg(ppm: float) -> float:
    """CO₂ 浓度 (ppm) → 分压 (mmHg)，标准大气压

    与固件 ai_bridge.c 的 AB_PPM_TO_MMHG_SCALE 一致，两端同源。
    上位机的特征提取要求输入 mmHg（寻峰门限 height=12 / prominence=8
    均为 mmHg 阈值），故收到 ppm 后必须做这一步换算。
    """
    return ppm * PPM_TO_MMHG


def mmhg_to_ppm(mmhg: float) -> float:
    """分压 (mmHg) → CO₂ 浓度 (ppm)，仅用于仿真数据生成"""
    return mmhg / PPM_TO_MMHG


def frame_checksum(data: bytes) -> int:
    """帧校验和：byte0~14 累加取低 8 位"""
    total = 0
    for b in data[:CHECKSUM_INDEX]:
        total += b
    return total & 0xFF


def encode_frame(concentrations_ppm: list, ai_class: int = None, rr_per_min: float = 0.0,
                 overflow: int = 0) -> bytes:
    """编码一帧 64 字节数据（模拟 STM32 端发送格式）
    Args:
        concentrations_ppm: 5 个 CO₂ 浓度值 (ppm)，v2.3 起取代原电压字段
        ai_class:    AI 分类 0/1/2/3，None 表示无 AI 结果
        rr_per_min:  呼吸率（次/分）
        overflow:    采样队列累计丢弃点数（byte 16-19，大端），默认 0
    Returns:
        64 字节帧（byte0=帧头 0xFF, byte15=校验和）
    """
    packet = bytearray(FRAME_SIZE)
    packet[0] = FRAME_HEADER
    for j in range(SAMPLES_PER_FRAME):
        ppm_int = max(0, min(65535, int(concentrations_ppm[j])))
        packet[CONCENTRATION_START + j * 2] = (ppm_int >> 8) & 0xFF
        packet[CONCENTRATION_START + j * 2 + 1] = ppm_int & 0xFF

    if ai_class is not None:
        packet[AI_FLAG_INDEX] = AI_MAGIC
        packet[AI_CLASS_INDEX] = ai_class & 0xFF
        rr_x10 = max(0, min(65535, int(rr_per_min * 10)))
        packet[RR_INDEX] = (rr_x10 >> 8) & 0xFF
        packet[RR_INDEX + 1] = rr_x10 & 0xFF

    ovf = max(0, min(0xFFFFFFFF, int(overflow)))
    packet[OVERFLOW_INDEX] = (ovf >> 24) & 0xFF
    packet[OVERFLOW_INDEX + 1] = (ovf >> 16) & 0xFF
    packet[OVERFLOW_INDEX + 2] = (ovf >> 8) & 0xFF
    packet[OVERFLOW_INDEX + 3] = ovf & 0xFF

    packet[CHECKSUM_INDEX] = frame_checksum(bytes(packet))
    # 其余字节保持 0x00（STM32 端行为一致）
    return bytes(packet)


def decode_frame(data: bytes) -> dict | None:
    """解码一帧 64 字节数据
    Returns:
        dict {concentration_ppm, ai_valid, ai_class, rr_value, overflow}
        帧头不符 / 校验和不符 / 帧长不足 → None（视为无效帧，交由同步重找）

    concentration_ppm 为 byte 1-10 的 5 个 CO₂ 浓度（ppm，大端 uint16）。
    v2.3 起取代原电压字段——旧固件发的是 mV，本端会把它误读为 ppm，
    数值偏大但不会报错，故两端必须同步升级。

    overflow 为固件累计丢弃的采样点数（byte 16-19，大端）。
    注意这是**累计值**而非本帧增量；判断某段时间是否丢点需自行做差。
    """
    if len(data) < FRAME_SIZE:
        return None
    if data[0] != FRAME_HEADER:
        return None
    if frame_checksum(data) != data[CHECKSUM_INDEX]:
        return None

    concentration_ppm = []
    for i in range(SAMPLES_PER_FRAME):
        base = CONCENTRATION_START + i * 2
        if base + 1 < FRAME_SIZE:
            ppm = (data[base] << 8) | data[base + 1]
            concentration_ppm.append(float(ppm))
        else:
            concentration_ppm.append(0.0)

    ai_valid = (data[AI_FLAG_INDEX] == AI_MAGIC)
    ai_class = None
    rr_value = None
    if ai_valid:
        ai_class = data[AI_CLASS_INDEX]
        rr_x10 = (data[RR_INDEX] << 8) | data[RR_INDEX + 1]
        rr_value = rr_x10 / 10.0

    # 采样队列累计丢弃点数（byte 16-19，大端）。旧固件此区为 0，解析结果自然为 0。
    overflow = ((data[OVERFLOW_INDEX] << 24) |
                (data[OVERFLOW_INDEX + 1] << 16) |
                (data[OVERFLOW_INDEX + 2] << 8) |
                data[OVERFLOW_INDEX + 3])

    return {
        'concentration_ppm': concentration_ppm,
        'ai_valid':    ai_valid,
        'ai_class':    ai_class,
        'rr_value':    rr_value,
        'overflow':    overflow,
    }


def find_frame_sync(buffer_data: bytes) -> int:
    """在字节流中搜索有效帧头位置（FRAME_HEADER=0xFF 在 byte 0 处）

    从一帧中间接入 / 丢字节后，用 0xFF 帧头重新对齐。
    帧内数据字节也可能出现 0xFF，因此需配合校验和判定：
    找到候选 0xFF → 若其后凑满 64 字节且校验和通过，则确认是帧头。

    Returns:
        帧起始偏移量；找不到返回 -1
    """
    if len(buffer_data) < FRAME_SIZE:
        return -1
    # 在缓冲区前 FRAME_SIZE 范围内搜索候选帧头
    max_off = min(len(buffer_data) - FRAME_SIZE + 1, FRAME_SIZE)
    for offset in range(max_off):
        if buffer_data[offset] != FRAME_HEADER:
            continue
        # 候选帧头 → 校验和确认（防止数据区 0xFF 误判）
        candidate = bytes(buffer_data[offset:offset + FRAME_SIZE])
        if decode_frame(candidate) is not None:
            return offset
    # 没有找到通过校验的帧 → 返回 -1（调用方据此跳过/丢弃）
    return -1


# ── 串口后台 Worker（整个 QSerialPort 生命周期在后台线程，彻底避免主线程卡死）──
class SerialPermanentWorker(QObject):
    """在后台线程管理 QSerialPort 全部操作：open / read / write / close

    命令信号（主线程调用 .emit()）→ 通过 Qt 队列连接投递到后台线程执行。
    注意：命令必须是 Signal，不能用 @Slot 方法 + .emit() —— PySide6 中
    普通方法没有 emit()，那样写会抛 AttributeError 导致串口永远打不开。
    """
    connected = Signal(bool, str)       # success, error_message
    disconnected = Signal()
    data_received = Signal(bytes)       # 从串口读取的原始字节
    error_occurred = Signal(str)        # 串口运行时错误（断连等）
    stage_log = Signal(str)             # 后台线程阶段日志（回传主线程记录）
    # 命令信号（主线程 → 后台线程）
    do_open = Signal(str, int)          # 请求打开串口 (port, baud)
    do_close = Signal()                 # 请求关闭串口
    do_send = Signal(bytes)             # 请求发送数据

    def __init__(self):
        super().__init__()
        self._ser = None
        # 命令信号连接到实现槽（moveToThread 后自动走 QueuedConnection）
        self.do_open.connect(self._impl_open)
        self.do_close.connect(self._impl_close)
        self.do_send.connect(self._impl_send)

    @Slot(str, int)
    def _impl_open(self, port_name: str, baud: int):
        """后台线程中打开串口（open() 可能阻塞，主线程用看门狗超时兜底）"""
        try:
            self.stage_log.emit(f"尝试打开串口 {port_name} @ {baud} baud ...")
            self._ser = QSerialPort()
            self._ser.setPortName(port_name)
            self._ser.setBaudRate(baud)
            self._ser.setDataBits(QSerialPort.Data8)
            self._ser.setStopBits(QSerialPort.OneStop)
            self._ser.setParity(QSerialPort.NoParity)
            self._ser.setFlowControl(QSerialPort.NoFlowControl)
            self._ser.readyRead.connect(self._on_ready_read)
            # 监听串口运行时错误（设备拔出等）
            self._ser.errorOccurred.connect(self._on_error)
            if self._ser.open(QSerialPort.ReadWrite):
                self.stage_log.emit(f"串口 {port_name} 打开成功")
                self.connected.emit(True, "")
            else:
                err = self._ser.errorString()
                self.stage_log.emit(f"串口 {port_name} 打开失败: {err}")
                self._ser = None
                self.connected.emit(False, err if err else "未知错误")
        except Exception as e:
            self.stage_log.emit(f"串口打开异常: {e}")
            self._ser = None
            self.connected.emit(False, str(e))

    @Slot()
    def _impl_close(self):
        """后台线程中关闭串口"""
        try:
            if self._ser is not None:
                if self._ser.isOpen():
                    self._ser.close()
                self._ser = None
        except Exception:
            self._ser = None
        self.disconnected.emit()

    @Slot(bytes)
    def _impl_send(self, data: bytes):
        """后台线程中发送数据到串口"""
        try:
            if self._ser is not None and self._ser.isOpen():
                written = self._ser.write(data)
                if written < 0:
                    self.error_occurred.emit(f"发送失败: {self._ser.errorString()}")
        except Exception as e:
            self.error_occurred.emit(f"发送异常: {e}")

    def _on_ready_read(self):
        """后台线程 readyRead 回调，非阻塞读取数据并转发到主线程"""
        try:
            if self._ser is not None:
                data = self._ser.readAll()
                if data.size() > 0:
                    self.data_received.emit(bytes(data.data()))
        except Exception:
            pass

    def _on_error(self, error):
        """后台线程串口错误回调"""
        err_map = {
            QSerialPort.NoError: None,
            QSerialPort.DeviceNotFoundError: "设备未找到",
            QSerialPort.PermissionError: "权限不足",
            QSerialPort.OpenError: "无法打开",
            QSerialPort.NotOpenError: "串口未打开",
            QSerialPort.WriteError: "写入失败",
            QSerialPort.ReadError: "读取失败",
            QSerialPort.ResourceError: "设备已断开",
            QSerialPort.UnsupportedOperationError: "不支持的操作",
            QSerialPort.TimeoutError: "操作超时",
            QSerialPort.UnknownError: "未知错误",
        }
        msg = err_map.get(error, f"串口错误码 {error}")
        if msg:
            self.error_occurred.emit(msg)


class CO2MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CO2 呼吸波形监控 - AI 智能诊断")
        self.resize(1000, 750)

        # ── 初始化数据库 / 配置 / 日志 ──
        self.db = DatabaseManager("co2_monitor.db")
        self.config = ConfigManager("config.json")
        self.logger = LogManager("logs")
        # 启动时清理过期数据（默认保留 30 天，可配置）
        retention = self.config.get("app.db_retention_days", 30)
        self.db.cleanup_old_data(retention_days=retention)
        self.logger.debug(f"已执行数据清理：保留最近 {retention} 天", source="system")
        self._alarm_consecutive = 0       # 连续异常计数
        self._last_ai_label = None        # 上一次 AI 标签
        self._last_alarm_time = None      # 上次告警通知时间（用于限频）
        self._db_write_counter = 0        # DB 写入节流计数器
        self._alert_flashing = False      # 状态栏告警闪烁状态
        self._alert_flash_timer = QTimer()  # 告警闪烁定时器
        self._alert_flash_timer.timeout.connect(self._flash_statusbar)
        self._trend_sample_count = 0      # 趋势分析采样计数
        self._auto_refresh_ticks = 0      # 历史表格自动刷新限频计数
        self._playback_rows = []          # 回放数据（正序）
        self._playback_playing = False    # 回放播放状态
        self._playback_pos = 0            # 回放当前进度索引
        self._playback_timer = QTimer()   # 回放定时器
        self._playback_timer.timeout.connect(self._playback_tick)
        # Stage 3 relative mode / event engine / isolated replay
        self.display_mode = DISPLAY_MODE_RELATIVE
        self._calibration_status = "uncalibrated"
        self._session_id = None
        self._stage3 = None
        self._stage3_alerted = set()
        self._replay_pack = None
        self._replay_session = None
        self._replay_runtime = None
        self._replay_timer = QTimer()
        self._replay_timer.setInterval(100)
        self._replay_timer.timeout.connect(self._replay_tick)
        self._open_watchdog = QTimer()    # 串口打开看门狗：open() 阻塞超时后强断重建
        self._open_watchdog.setSingleShot(True)
        self._open_watchdog.setInterval(10000)   # 10 秒
        self._open_watchdog.timeout.connect(self._on_open_timeout)

        self.load_onnx_model()       # ← 加载你训练的 Random Forest 模型
        self.setup_ui()
        self._setup_tray_icon()            # 系统托盘图标
        self.logger.set_ui_callback(self._log_to_ui)  # 日志同步到 GUI
        self.setup_serial()
        self.setup_data_buffers()
        self._init_stage3()
        self.setup_timer()
        self.setup_simulator()

        self.logger.info("系统启动完成", source="system")
        self.logger.debug(f"DB={self.db._db_path}  Config={self.config._config_path}", source="system")

    # ═══════════════════════════════════════════
    #  ONNX 模型加载
    # ═══════════════════════════════════════════

    def load_onnx_model(self):
        """加载 train_model.py 训练并导出的 ONNX 模型"""
        if os.path.exists(MODEL_PATH):
            try:
                self.ort_session = ort.InferenceSession(MODEL_PATH)
                self.ort_input_name = self.ort_session.get_inputs()[0].name
                self.ort_output_names = [o.name for o in self.ort_session.get_outputs()]
                self.model_loaded = True
                self.logger.info(f"ONNX 模型已加载: {os.path.basename(MODEL_PATH)}", source="ai")
                self.logger.debug(f"输入={self.ort_input_name}, 输出={self.ort_output_names}", source="ai")
            except Exception as e:
                self.logger.warn(f"ONNX 模型加载失败: {e}", source="ai")
                self.model_loaded = False
        else:
            self.logger.warn(f"未找到模型文件: {MODEL_PATH}", source="ai")
            self.model_loaded = False

    # ═══════════════════════════════════════════
    #  UI 搭建
    # ═══════════════════════════════════════════


    # ═══════════════════════════════════════════
    #  Stage 3 相对模式 / 事件引擎
    # ═══════════════════════════════════════════

    def set_display_mode(self, mode: str):
        """Only relative mode is available; calibration records cannot unlock C."""
        requested = str(mode).lower()
        if requested != DISPLAY_MODE_RELATIVE:
            self.log_msg(
                f"拒绝切换到 {mode}：未标定设备 C 模式继续禁用；保持相对模式",
                level="warn", source="mode")
        self.display_mode = DISPLAY_MODE_RELATIVE
        if hasattr(self, "mode_label"):
            self.mode_label.setText("相对模式（未标定）")
        self._reset_stage3_session("模式切换/保持相对模式")

    def _init_stage3(self):
        self._stage3 = Stage3Runtime()
        if self._stage3.available:
            self.log_msg("Stage 3 U15/U30 事件引擎已加载（相对模式）", source="ai")
            try:
                self.model_status_label.setText("Stage3 U15/U30 已加载")
                self.model_status_label.setStyleSheet(
                    "color:#00CC66; font-size:10px;")
            except Exception:
                pass
        else:
            self.log_msg(f"Stage 3 事件引擎不可用: {self._stage3.error}",
                         level="warn", source="ai")
        self._reset_stage3_session("初始化")

    def _reset_stage3_session(self, reason: str = ""):
        if self._stage3 is not None and self._stage3.available:
            # Close any open output at the expected decision deadline before
            # discarding history.  This is the same interruption rule used by
            # offline evaluation and avoids silently losing an ACTIVE event.
            try:
                transitions = self._stage3.interrupt(
                    reason=reason or "SESSION_RESET")
                if transitions:
                    self._handle_stage3_transitions(transitions)
            except Exception:
                pass
            try:
                self._stage3.reset()
            except Exception:
                pass
        self._stage3_alerted.clear()
        self._session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self._last_stage3_states = {}
        if hasattr(self, "event_label"):
            self.event_label.setText(
                "事件引擎：等待数据 · 呼气 CO2 低值/消失样事件")
            self.event_label.setStyleSheet(
                "color:#AAAAAA; background-color:#202020; "
                "border-radius:6px; padding:6px;")
        if reason:
            self.log_msg(f"Stage 3 会话清空: {reason}", source="event")

    def _legacy_alarm_enabled(self) -> bool:
        """Relative mode deliberately keeps the legacy firmware alarm chain off."""
        return False

    def _handle_stage3_transitions(self, transitions):
        if not transitions or self._stage3 is None:
            return
        for tr in transitions:
            kind = tr.get("kind")
            scale = tr.get("scale", "")
            t = float(tr.get("time", 0.0))
            eid = tr.get("episode_id")
            if kind in ("SUSPECT_START", "ACTIVE_START",
                        "ACTIVE_END", "SUSPECT_END"):
                # Persist from the transition snapshot.  Do not query the
                # current engine state: an ending transition has already
                # cleared the episode and would lose onset/first-active times.
                ep = tr.get("episode") or {}
                end_time = None
                if kind == "SUSPECT_END":
                    end_time = ep.get("end_time")
                elif kind == "ACTIVE_END":
                    intervals = ep.get("active_intervals") or []
                    end_time = (intervals[-1].get("end") if intervals
                                else ep.get("end_time"))
                try:
                    self.db.insert_event_record(
                        ts=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        session_id=self._session_id,
                        mode=self.display_mode,
                        scale=scale,
                        state=kind,
                        label=EVENT_LABEL,
                        estimated_onset=ep.get("estimated_onset"),
                        first_prompt_time=ep.get("first_prompt_time"),
                        first_active_time=ep.get("first_active_time"),
                        end_time=end_time,
                        source="stage3_event_engine",
                        episode_id=ep.get("episode_id") or eid,
                        transition_seq=tr.get("transition_seq"))
                except Exception:
                    pass
                self.log_msg(
                    f"[事件] {scale} {kind} t={t:.1f}s ({EVENT_LABEL})",
                    level="warn" if kind == "ACTIVE_START" else "info",
                    source="event")
            if kind == "ACTIVE_START":
                key = (scale, eid)
                if key not in self._stage3_alerted:
                    self._stage3_alerted.add(key)
                    self._trigger_event_alert(scale, t)
        self._update_stage3_label()

    def _update_stage3_label(self):
        if not hasattr(self, "event_label"):
            return
        if self._stage3 is None or not self._stage3.available:
            self.event_label.setText(
                "事件引擎不可用：相对模式仍隐藏绝对浓度；不触发旧告警")
            self.event_label.setStyleSheet(
                "color:#FF9933; background-color:#202020; "
                "border-radius:6px; padding:6px;")
            return
        st = self._stage3.states()
        parts = []
        any_active = False
        any_suspect = False
        any_unknown = False
        for s in ("U15", "U30"):
            state = st.get(s, {}).get("state", "WAITING")
            parts.append(f"{s}: {state}")
            any_active |= state == "ACTIVE"
            any_suspect |= state == "SUSPECT"
            any_unknown |= state == "UNKNOWN"
        color = "#FF4444" if any_active else (
            "#FF9933" if any_suspect else ("#AAAAAA" if any_unknown else "#00CC66"))
        self.event_label.setText(
            "相对模式 · 事件引擎 | " + " | ".join(parts) + "\n"
            "定义：呼气 CO2 低值/消失样事件（非临床确诊屏息）")
        self.event_label.setStyleSheet(
            f"color:{color}; background-color:#202020; "
            "border-radius:6px; padding:6px;")

    def _trigger_event_alert(self, scale: str, t: float):
        """Only Stage 3 ACTIVE uses the user notification chain."""
        msg = f"{scale} ACTIVE @{t:.1f}s · {EVENT_LABEL}"
        try:
            self._tray.showMessage(
                "相对模式呼吸事件", msg,
                QSystemTrayIcon.Warning, 5000)
            if self.config.get("alarm.sound_enabled", False):
                QApplication.beep()
        except Exception:
            pass
        self.log_msg(f"⚠ {msg}", level="warn", source="event")

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── 第1行：控制栏 ──
        row1 = QHBoxLayout()

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(100)
        self.refresh_ports()

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "256000"])
        saved_baud = str(self.config.get("serial.baud_rate", 115200))
        self.baud_combo.setCurrentText(saved_baud)

        self.connect_btn = QPushButton("打开串口")
        self.connect_btn.setFixedWidth(80)
        self.connect_btn.clicked.connect(self.toggle_serial)

        self.conn_status = QLabel("● 未连接")
        self.conn_status.setStyleSheet("color:#888888; font-weight:bold;")

        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(50)
        refresh_btn.clicked.connect(self.refresh_ports)

        # 模拟按钮
        self.sim_btn = QPushButton("▶ 启动模拟")
        self.sim_btn.setFixedWidth(90)
        self.sim_btn.clicked.connect(self.toggle_simulator)
        self.sim_btn.setStyleSheet(
            "QPushButton { background-color:#FF6600; color:white; font-weight:bold; "
            "border-radius:4px; padding:4px; }"
        )

        # 清除按钮
        self.clear_btn = QPushButton("清除")
        self.clear_btn.setFixedWidth(50)
        self.clear_btn.clicked.connect(self.clear_simulator)
        self.clear_btn.setEnabled(False)

        self.sim_pattern_label = QLabel("")
        self.sim_pattern_label.setStyleSheet("color:#FF9933; font-weight:bold;")

        # 模型状态标签
        model_status = "ONNX(旧)已加载" if self.model_loaded else "ONNX(旧)未找到"
        model_color = "#00CC66" if self.model_loaded else "#FF4444"
        self.model_status_label = QLabel(model_status)
        self.model_status_label.setStyleSheet(f"color:{model_color}; font-size:10px;")

        row1.addWidget(QLabel("串口:"))
        row1.addWidget(self.port_combo)
        row1.addWidget(QLabel("波特率:"))
        row1.addWidget(self.baud_combo)
        row1.addWidget(self.connect_btn)
        row1.addWidget(refresh_btn)
        row1.addWidget(self.conn_status)
        self.mode_label = QLabel("相对模式（未标定）")
        self.mode_label.setStyleSheet(
            "color:#FFDD44; font-weight:bold; border:1px solid #666; "
            "border-radius:4px; padding:2px 6px;")
        row1.addWidget(self.mode_label)
        row1.addSpacing(15)
        row1.addWidget(self.sim_btn)
        row1.addWidget(self.clear_btn)
        row1.addWidget(self.sim_pattern_label)
        row1.addStretch()
        row1.addWidget(self.model_status_label)
        # 控制栏常驻顶部
        layout.addLayout(row1)

        # ── 顶层双 Tab：实时监控 / 历史数据 ──
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #555; }")

        # ═══ 页1：实时监控 ═══
        monitor_page = QWidget()
        monitor_layout = QVBoxLayout(monitor_page)
        monitor_layout.setSpacing(5)
        monitor_layout.setContentsMargins(4, 4, 4, 4)

        # ── 数值仪表盘 ──
        # v2.3 起固件帧内直接下发浓度，不再有原始传感器电压，故取消该卡片
        row2 = QHBoxLayout()
        self.co2_value   = self._make_dash_card("实时 CO₂（相对）", "— 未标定", "#00AA00")
        self.rr_value    = self._make_dash_card("呼吸率", "-- 次/分", "#3399FF")
        self.etco2_value = self._make_dash_card("EtCO₂（未标定）", "—", "#FF9933")
        row2.addWidget(self.co2_value)
        row2.addWidget(self.rr_value)
        row2.addWidget(self.etco2_value)
        monitor_layout.addLayout(row2)

        # ── AI 诊断结果 ──
        self.ai_label = QLabel("点击「启动模拟」查看效果")
        self.ai_label.setAlignment(Qt.AlignCenter)
        self.ai_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.ai_label.setMinimumHeight(40)
        self.ai_label.setStyleSheet(
            "color:#FFFFFF; background-color:#2D2D2D; "
            "border-radius:8px; padding:6px;"
        )
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Microsoft YaHei", 11))
        self.confidence_label.setStyleSheet("color:#AAAAAA;")
        self.event_label = QLabel("事件引擎：等待数据 · 呼气 CO2 低值/消失样事件")
        self.event_label.setWordWrap(True)
        self.event_label.setStyleSheet(
            "color:#AAAAAA; background-color:#202020; border-radius:6px; padding:6px;")
        monitor_layout.addWidget(self.ai_label)
        monitor_layout.addWidget(self.confidence_label)
        monitor_layout.addWidget(self.event_label)

        # ── 波形 + 日志区 ──
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.addWidget(QLabel("相对 CO₂ 波形（未标定，仅形态）"))
        self.waveform = pg.PlotWidget()
        self.waveform.setLabel('left', '相对 CO₂')
        self.waveform.setLabel('bottom', '时间', units='秒')
        self.waveform.showGrid(x=True, y=True, alpha=0.3)
        self.waveform.setYRange(0, 60)
        self.waveform.setBackground('#1A1A1A')
        try:
            self.waveform.getAxis('left').setStyle(showValues=False)
        except Exception:
            pass
        self.curve = self.waveform.plot(pen=pg.mkPen(color=(0, 255, 100), width=1.5))

        # ── 鼠标交互：左键拖拽平移，右键框选缩放，滚轮缩放 ──
        vb = self.waveform.getViewBox()
        vb.setMouseMode(vb.PanMode)  # 三键模式：左键=平移，右键=框选缩放
        vb.setLimits(yMin=-5, yMax=100)
        self.auto_scroll = True   # 运行中自动滚动；停止后用户可自由拖拽

        # 波形上方的提示标签
        self.scroll_hint = QLabel("运行中：自动滚动 | 停止后可拖拽查看")
        self.scroll_hint.setStyleSheet("color:#888888; font-size:10px;")

        lv.addWidget(self.waveform)
        lv.addWidget(self.scroll_hint)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("数据日志"))
        self.packet_count_label = QLabel("接收包: 0")
        self.packet_count_label.setStyleSheet("color:#888888;")
        top_bar.addStretch()
        top_bar.addWidget(self.packet_count_label)
        self.log = QPlainTextEdit()
        clear_btn = QPushButton("清除")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self.log.clear)
        top_bar.addWidget(clear_btn)
        rv.addLayout(top_bar)
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Microsoft YaHei", 9))
        self.log.setMaximumBlockCount(500)
        rv.addWidget(self.log)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([650, 350])
        splitter.setMinimumHeight(200)  # 波形区保底高度
        monitor_layout.addWidget(splitter, stretch=3)

        self.main_tabs.addTab(monitor_page, "实时监控")

        # ═══ 页2：历史数据 ═══
        self._setup_history_tab()
        history_page = QWidget()
        hl = QVBoxLayout(history_page)
        hl.setContentsMargins(4, 4, 4, 4)
        hl.addWidget(self._history_tabs)
        self.main_tabs.addTab(history_page, "历史数据")

        # 顶层 tab 切换时刷新历史
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)

        layout.addWidget(self.main_tabs)

        self.statusBar().showMessage(
            "就绪 — 点击「启动模拟」查看效果（ONNX模型推理），或连接STM32串口"
        )
        # 点击状态栏停止告警闪烁
        self.statusBar().mousePressEvent = lambda e: self._stop_alert()

    def _make_dash_card(self, title, value, color):
        card = QFrame()
        card.setStyleSheet("QFrame { background-color:#2D2D2D; border-radius:6px; padding:4px; }")
        box = QVBoxLayout(card)
        box.setSpacing(2)
        tl = QLabel(title)
        tl.setStyleSheet("color:#AAAAAA; font-size:10px;")
        tl.setAlignment(Qt.AlignCenter)
        vl = QLabel(value)
        vl.setStyleSheet(f"color:{color}; font-size:17px; font-weight:bold;")
        vl.setAlignment(Qt.AlignCenter)
        vl.setObjectName(f"val_{title}")
        box.addWidget(tl)
        box.addWidget(vl)
        return card

    # ═══════════════════════════════════════════
    #  串口（全部在后台线程操作，避免主线程卡死）
    # ═══════════════════════════════════════════

    def setup_serial(self):
        """创建后台线程 + SerialPermanentWorker，串口操作全部在后台"""
        self._serial_is_open = False          # 主线程追踪串口状态
        self._serial_opening = False          # 打开进行中（后台 open 可能阻塞，允许取消）
        self._serial_thread = QThread()
        self._serial_worker = SerialPermanentWorker()
        self._serial_worker.moveToThread(self._serial_thread)

        # 连接成功/失败 → 更新 UI
        self._serial_worker.connected.connect(self._on_serial_connected)
        # 串口已关闭
        self._serial_worker.disconnected.connect(self._on_serial_disconnected)
        # 后台读取到的数据 → 主线程解析
        self._serial_worker.data_received.connect(self._on_serial_bytes)
        # 串口运行时错误 → 日志
        self._serial_worker.error_occurred.connect(
            lambda msg: self.log_msg(f"[串口错误] {msg}")
        )
        # 后台线程阶段日志 → 日志文件（排查 open() 卡住）
        self._serial_worker.stage_log.connect(
            lambda msg: self.log_msg(msg, source="serial")
        )

        self._serial_thread.start()

    def refresh_ports(self):
        self.port_combo.clear()
        for info in QSerialPortInfo.availablePorts():
            name = info.portName()
            if name:
                self.port_combo.addItem(name)

    def _on_open_timeout(self):
        """看门狗：后台 open() 10 秒未返回 → 判定阻塞，强制断开并恢复界面"""
        if not self._serial_opening:
            return
        self.log_msg("[WARN] 串口打开超时（10秒），可能端口被占用或设备无响应，正在强制中断...", source="serial")
        self.statusBar().showMessage("串口打开超时，已自动取消")
        self._serial_opening = False
        self._serial_is_open = False
        self.connect_btn.setText("打开串口")
        self.conn_status.setText("● 未连接")
        self.conn_status.setStyleSheet("color:#888888; font-weight:bold;")
        # 后台线程可能卡在 open() 上，无法处理信号 → 整体重建线程
        self._teardown_serial_backend()
        self.setup_serial()
        self.refresh_ports()

    def _on_serial_connected(self, success: bool, error_msg: str):
        """后台 open() 完成后的回调（主线程）"""
        self._serial_opening = False
        self._open_watchdog.stop()

        if success:
            self._serial_is_open = True
            self.connect_btn.setText("关闭串口")
            self.conn_status.setText("● 已连接")
            self.conn_status.setStyleSheet("color:#00CC66; font-weight:bold;")
            self.auto_scroll = True
            self.scroll_hint.setText("串口已连接 | 实时自动滚动中")
            # 新病例/新会话：不得继承上一会话的窗口、质量历史或事件证据
            self.wave_buffer.clear()
            self.wave_timestamps.clear()
            self._reset_ai_window()
            self._reset_stage3_session("串口连接/病例切换")
            port = self.port_combo.currentText()
            baud = int(self.baud_combo.currentText())
            self.statusBar().showMessage(
                f"串口 {port} 已连接（后台线程）")
            self.log_msg(f"[串口] {port} 已连接", source="serial")
            # 保存串口参数到配置文件
            self.config.set("serial.port", port)
            self.config.set("serial.baud_rate", baud)
            self.config.save()
        else:
            self._serial_is_open = False
            self.connect_btn.setText("打开串口")
            self.conn_status.setText("● 未连接")
            self.conn_status.setStyleSheet("color:#888888; font-weight:bold;")
            self.log_msg(f"[ERROR] 串口打开失败: {error_msg}")
            self.statusBar().showMessage(f"串口不可用: {error_msg}")

    def _on_serial_disconnected(self):
        """后台 close() 完成后的回调（主线程）"""
        self._serial_is_open = False
        self._serial_opening = False
        self.connect_btn.setText("打开串口")
        self.conn_status.setText("● 未连接")
        self.conn_status.setStyleSheet("color:#888888; font-weight:bold;")
        self.auto_scroll = False
        self.scroll_hint.setText("串口已断开 | 可拖拽查看已接收的波形")
        self._reset_stage3_session("串口断开")
        self.statusBar().showMessage("串口已断开 — 可拖拽查看历史波形")

    def _on_serial_bytes(self, data: bytes):
        """后台线程读取到的原始字节 → 主线程拼包解析（含帧同步恢复）

        用帧头 0xFF + 校验和定位帧边界：从一帧中间接入/丢字节后，
        逐字节滑动直到找到"帧头+校验通过"的完整帧，彻底避免错拼电压。
        """
        try:
            self.serial_buffer.extend(data)
            while len(self.serial_buffer) >= FRAME_SIZE:
                offset = find_frame_sync(bytes(self.serial_buffer))
                if offset < 0:
                    # 前 FRAME_SIZE 内无有效帧 → 丢弃 1 字节继续滑（错位数据）
                    self.serial_buffer = self.serial_buffer[1:]
                    continue
                if offset > 0:
                    self.log_msg(f"[串口] 字节丢步，跳过 {offset} 字节以同步帧头")
                    self.serial_buffer = self.serial_buffer[offset:]
                    if len(self.serial_buffer) < FRAME_SIZE:
                        break
                # offset==0：缓冲区已对齐到帧头，直接切帧
                frame = decode_frame(bytes(self.serial_buffer[:FRAME_SIZE]))
                self.serial_buffer = self.serial_buffer[FRAME_SIZE:]
                if frame is not None:
                    self._apply_frame(frame)
        except Exception as e:
            self.log_msg(f"[ERROR] 串口数据解析异常: {e}")

    def _compute_rr_from_buffer(self) -> float | None:
        """从波形缓冲实时计算呼吸率（次/分）— 帧未带 RR 时使用
        用与 STM32 一致的上升沿-波峰状态机寻峰（抗噪声，不误判肩部）
        """
        buf = list(self.wave_buffer)
        n = len(buf)
        if n < 200:
            return None

        # 状态机寻峰：0=找波峰(上升), 1=找波谷(下降)
        peak_count = 0
        last_peak_idx = -100
        min_distance = int(0.8 * 25)  # 0.8s × 25Hz
        state = 0
        current_max = buf[0]
        current_max_idx = 0
        current_min = buf[0]

        for i in range(1, n):
            if state == 0:  # 上升段找波峰
                if buf[i] > current_max:
                    current_max = buf[i]
                    current_max_idx = i
                if current_max - buf[i] >= 8.0:  # 从峰值回落 8 mmHg
                    if current_max > 10.0 and (current_max_idx - last_peak_idx) >= min_distance:
                        peak_count += 1
                        last_peak_idx = current_max_idx
                    state = 1
                    current_min = buf[i]
            else:  # 下降段找波谷
                if buf[i] < current_min:
                    current_min = buf[i]
                if buf[i] - current_min >= 8.0 or buf[i] < 10.0:
                    state = 0
                    current_max = buf[i]
                    current_max_idx = i

        duration_min = n * 0.040 / 60.0  # 每点 40ms
        if duration_min <= 0:
            return None
        return round(peak_count / duration_min, 1)

    def _apply_frame(self, frame: dict):
        """将解码后的帧数据应用到波形缓冲、仪表盘、数据库和告警检测"""
        self.packet_count += 1

        concentration_ppm = frame['concentration_ppm']
        ai_valid = frame['ai_valid']
        ai_class = frame['ai_class']
        rr_value = frame['rr_value']
        confidence = None  # 初始化，供下方 DB 写入和告警检测使用

        # ── 采样队列溢出检查 ──
        # 固件以精确 25Hz 在中断里采样、主循环异步消费；队列满时会丢弃采样点
        # 并累加计数（帧字节 16-19）。累计值单调递增，本帧增量 > 0 即表示
        # 自上一帧以来有丢点 —— 那段波形存在缺口，据其得出的结论不可信。
        overflow_now = frame.get('overflow', 0)
        if self._last_overflow is not None and overflow_now > self._last_overflow:
            dropped = overflow_now - self._last_overflow
            self.overflow_total += dropped
            self.overflow_warning = True
            # 丢点造成数据中断：事件证据不得跨缺口延续
            self._reset_stage3_session("采样队列溢出/数据中断")
            # 累计超过阈值（约 2.5 秒采样）时留痕，便于事后排查
            if self.overflow_total >= OVERFLOW_ALERT_POINTS:
                self.log_manager.warn(
                    f"采样队列溢出累计 {self.overflow_total} 点，波形存在缺口")
        self._last_overflow = overflow_now

        now = datetime.now()
        ts_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
        if self.start_time is None:
            self.start_time = now
        elapsed = (now - self.start_time).total_seconds() - self.total_pause_duration

        # 波形缓冲 + AI 推理缓冲（模拟+硬件共用，统一在这里喂入）
        # 帧内是固件算好的 ppm；换算成 mmHg 后喂入，供显示与特征提取使用
        # （特征提取的寻峰门限是 mmHg 阈值，必须换算）。
        co2_values = []
        for j, ppm in enumerate(concentration_ppm):
            co2 = ppm_to_mmhg(ppm)
            co2_values.append(co2)
            self.wave_buffer.append(co2)
            self.ai_data_buffer.append(co2)
            pt = elapsed - 0.200 + (j * 0.040)
            self.wave_timestamps.append(pt)

        # Stage 3 relative-mode event engine (U15/U30 independent)
        try:
            transitions = self._stage3.feed_samples(co2_values)                 if self._stage3 is not None else []
            self._handle_stage3_transitions(transitions)
        except Exception as exc:
            self.log_msg(f"[事件引擎] {exc}", level="error", source="event")

        # 仪表盘：相对模式不显示绝对 mmHg/EtCO2
        latest_ppm = concentration_ppm[-1] if concentration_ppm else 0.0
        latest_co2 = ppm_to_mmhg(latest_ppm)
        if self.display_mode == DISPLAY_MODE_RELATIVE:
            self._update_card("实时 CO₂（相对）", "— 未标定", "#888888")
            self._update_card("EtCO₂（未标定）", "—", "#888888")
        else:
            self._update_card("实时 CO₂（相对）", f"{latest_co2:.1f} mmHg", "#00AA00")

        # ── 数据库写入（每 5 包写一次，减少 I/O）──
        self._db_write_counter += 1
        if self._db_write_counter % 5 == 0:
            etco2_val = None
            if len(self.wave_buffer) > 10:
                peaks = []
                buf = list(self.wave_buffer)
                for i in range(1, len(buf) - 1):
                    if buf[i] > buf[i - 1] and buf[i] > buf[i + 1] and buf[i] > 10:
                        peaks.append(buf[i])
                if peaks:
                    etco2_val = max(peaks[-5:]) if len(peaks) >= 5 else max(peaks)

            # 帧未带呼吸率时，从波形缓冲实时计算（避免 DB 大量 NULL）
            if rr_value is None:
                rr_value = self._compute_rr_from_buffer()

            confidence = None
            if ai_valid and ai_class is not None and hasattr(self, 'latest_probs') and self.latest_probs is not None:
                # latest_probs 是 ZipMap 概率字典（键 0/1/2/3），用 .get 读，缺键视为高置信度
                confidence = float(self.latest_probs.get(ai_class, 1.0))

            # voltage_v 列自 v2.3 起不再有电压可存（帧内已改为浓度），
            # 保留写入 ppm 以便追溯原始读数；列名维持不变以免影响既有库。
            rel = self.display_mode == DISPLAY_MODE_RELATIVE
            self.db.insert_device_data(
                ts=ts_iso,
                co2=None if rel else latest_co2,
                rr=rr_value,
                etco2=None if rel else etco2_val,
                voltage_v=None if rel else latest_ppm,
                ai_label=ai_class if (ai_valid and not rel) else None,
                confidence=None if rel else confidence,
            )

        # AI 诊断
        if ai_valid and ai_class is not None:
            label_name = LABEL_MAP.get(ai_class, f"未知({ai_class})")
            color = LABEL_COLORS.get(ai_class, "#FFFFFF")

            # 区分结论来源：端侧固件推理 / 上位机本地推理
            # 固件在字节 11-14 给出标签与呼吸率，是产品的真实能力；
            # 上位机本地推理（每 30 秒一次）仅用于交叉验证。
            self.latest_fw_label = ai_class
            prefix = "[旧固件分类·仅参考]"

            # 组装诊断文本：基础结论 + 可选的"预热中"与"本地不一致"标注
            text = f"{prefix} AI 诊断: {label_name}"
            tips = []

            # 预热中标注：窗口未满 90s 时结论不可靠（急促易误判为正常），
            # 明确提示用户仅作参考，避免被早期错误结论误导。
            warmup = self._is_warming_up()
            if warmup:
                filled = int(self._window_fill_ratio() * AI_WINDOW_SECONDS)
                text += f"  ⏳ 预热中 {filled}/{int(AI_WINDOW_SECONDS)}s"
                tips.append(
                    f"推理需要 {int(AI_WINDOW_SECONDS)} 秒连续数据，当前仅采集 {filled} 秒。\n"
                    "预热期间的诊断结果仅供参考，不作为告警依据。")

            # 端云交叉验证：本地推理与端侧结论不一致时提示
            # 两者独立实现（端侧 C / 上位机 Python），一致即互为佐证；
            # 不一致说明该窗口的判定处于模型边界，临床解读需谨慎。
            local_lab = getattr(self, 'latest_local_label', None)
            if local_lab is not None and local_lab != ai_class:
                local_name = LABEL_MAP.get(local_lab, f"未知({local_lab})")
                text += f"  ⚠ 本地={local_name}"
                tips.append(
                    "端侧固件与上位机本地推理结论不一致。"
                    "两条链路独立实现，不一致通常说明该窗口处于模型判别边界。")

            self.ai_label.setText(text)
            self.ai_label.setToolTip("\n".join(tips))
            self.ai_label.setStyleSheet(
                f"color:#FFFFFF; background-color:{color}; "
                f"border-radius:8px; padding:10px; font-weight:bold;"
            )

            conf_str = ""
            if hasattr(self, 'latest_probs') and self.latest_probs is not None:
                conf_str = f" 置信度={self.latest_probs.get(ai_class, 1.0):.0%}"
            elif self.overflow_warning:
                conf_str = "  ⚠ 波形存在缺口"

            if rr_value is not None:
                self.confidence_label.setText(
                    f"分类ID={ai_class} | 呼吸率={rr_value:.1f} 次/分{conf_str}"
                )
                self._update_card("呼吸率", f"{rr_value:.1f} 次/分", "#3399FF")

            # ── 告警检测 ──
            alarm_enabled = (self.config.get("alarm.enabled", True)
                             and self._legacy_alarm_enabled())
            if alarm_enabled:
                abnormal_labels = {1, 2, 3}  # 非"正常"即为异常
                min_confidence = self.config.get("alarm.min_confidence", 0.6)
                consecutive_threshold = self.config.get("alarm.consecutive_threshold", 2)

                if ai_class in abnormal_labels:
                    if ai_class == self._last_ai_label:
                        self._alarm_consecutive += 1
                    else:
                        self._alarm_consecutive = 1
                        self._last_ai_label = ai_class

                    # 无概率数据(STM32帧/模拟真值)时视为高置信度，不拦截告警
                    # 只有确有待检概率时才应用 min_confidence 门槛
                    current_conf = confidence if confidence is not None else 1.0
                    passes_confidence = (confidence is None) or (current_conf >= min_confidence)
                    # 预热期（窗口未满 90s）诊断不可靠 → 不记录、不告警
                    passes_warmup = not self._is_warming_up()
                    # 冷却：告警检测逐帧运行，持续异常时会每 0.4 秒触发一次，
                    # 导致 DB 记录与弹窗刷屏。同一冷却期内只记录并提示一次。
                    # 注意：被冷却抑制时不重置计数，因此异常持续期间冷却到期会再次告警。
                    cooldown_ok = (
                        self._last_alarm_time is None
                        or (now - self._last_alarm_time).total_seconds() >= ALARM_NOTIFY_COOLDOWN_SEC
                    )
                    if (self._alarm_consecutive >= consecutive_threshold
                            and passes_confidence and passes_warmup and cooldown_ok):
                        alarm_types = {1: "急促呼吸", 2: "屏息", 3: "呼吸异常"}
                        alarm_type = alarm_types.get(ai_class, f"未知({ai_class})")
                        self.db.insert_alarm(
                            ts=ts_iso, alarm_type=alarm_type,
                            ai_label=ai_class, confidence=current_conf,
                            co2_value=latest_co2, rr_value=rr_value,
                        )
                        # 通知用户（含弹窗/蜂鸣，内部同样受冷却保护）
                        self._trigger_alert(alarm_type, ai_class, latest_co2, rr_value)
                        # 重置计数防止重复触发
                        self._alarm_consecutive = 0
                        self._last_ai_label = None
                else:
                    self._alarm_consecutive = 0
                    self._last_ai_label = None

        # ── 趋势预警：相对模式禁用绝对 EtCO2 阈值告警 ──
        if self._legacy_alarm_enabled():
            self._trend_sample_count += 1
            if self._trend_sample_count % 50 == 0:
                self._analyze_trends()

        # 日志（每 25 包）
        if self.packet_count % 25 == 0:
            ai_str = f" AI={LABEL_MAP.get(ai_class, '?')}" if ai_valid else ""
            if self.display_mode == DISPLAY_MODE_RELATIVE:
                self.log_msg(f"[{now:%H:%M:%S}] 包#{self.packet_count} | "
                             f"相对CO₂(未标定){ai_str}")
            else:
                self.log_msg(f"[{now:%H:%M:%S}] 包#{self.packet_count} | "
                             f"CO2={latest_co2:.1f}mmHg{ai_str}")

    def toggle_serial(self):
        """打开 / 关闭 / 取消串口 — 通过信号触发后台线程操作，主线程立即返回不阻塞"""
        if self._serial_is_open:
            # 已连接 → 关闭
            self.connect_btn.setText("打开串口")
            self.conn_status.setText("● 关闭中...")
            self.conn_status.setStyleSheet("color:#FF9933; font-weight:bold;")
            self._serial_worker.do_close.emit()
        elif self._serial_opening:
            # 打开进行中（后台可能卡住了）→ 取消此次打开
            self.connect_btn.setText("打开串口")
            self.conn_status.setText("● 取消中...")
            self.conn_status.setStyleSheet("color:#FF9933; font-weight:bold;")
            self.statusBar().showMessage("正在取消串口连接...")
            self.log_msg("[串口] 用户取消打开，正在关闭后台尝试...")
            self._serial_opening = False
            self._serial_worker.do_close.emit()
        else:
            port = self.port_combo.currentText()
            if not port:
                self.log_msg("[ERROR] 未选择串口，请先选择可用端口")
                self.statusBar().showMessage("未选择串口")
                return

            # 预检：端口不在枚举列表时仅提示、不拦截（兼容旧版直接 open 的行为；
            # 有些环境枚举异常/时序晚于点击，硬拒会导致"以前能连现在不能连"）
            avail = [info.portName() for info in QSerialPortInfo.availablePorts()]
            if port not in avail:
                self.log_msg(f"[WARN] 串口 {port} 不在当前枚举列表中，仍尝试打开（可能端口刚插入/驱动未刷新）",
                             source="serial")
                self.statusBar().showMessage(f"串口 {port} 未在列表中，仍尝试打开...")

            # 发起打开（按钮立即显示"关闭串口"，允许用户随时取消）
            self._serial_opening = True
            self.connect_btn.setText("关闭串口")
            self.conn_status.setText("● 连接中...")
            self.conn_status.setStyleSheet("color:#FF9933; font-weight:bold;")
            self.statusBar().showMessage(f"正在打开串口 {port}（后台线程）...")
            # 启动看门狗：10 秒内没收到 connected 回调 → 判定 open() 卡死，强断重建
            self._open_watchdog.start()
            baud = int(self.baud_combo.currentText())
            self._serial_worker.do_open.emit(port, baud)

    # ═══════════════════════════════════════════
    #  数据缓冲
    # ═══════════════════════════════════════════

    def setup_data_buffers(self):
        self.wave_buffer = deque(maxlen=7500)       # 最大 5 分钟数据供回看
        self.wave_timestamps = deque(maxlen=7500)
        self.serial_buffer = bytearray()
        self.packet_count = 0
        self.start_time = None
        self.total_pause_duration = 0.0
        self.pause_start = None
        self.ai_data_buffer = deque(maxlen=AI_WINDOW_POINTS)  # 90 秒 AI 推理窗口 (2250 点)
        self.auto_scroll = True
        # 最近一次推理结果（供 UI 持续显示，避免两次推理间显示消失）
        self.latest_probs = None
        self.latest_label = None
        self.latest_rr = 0.0

        # 采样队列溢出跟踪（帧字节 16-19，累计值）
        self._last_overflow = None    # 上一帧的累计值，None 表示尚未收到首帧
        self.overflow_total = 0       # 本次会话累计丢弃点数
        self.overflow_warning = False # 是否发生过溢出

        # 端侧与上位机结论对照（用于交叉验证提示）
        self.latest_fw_label = None   # 固件发来的标签
        self.latest_local_label = None  # 上位机本地推理标签

    def setup_timer(self):
        self.wave_timer = QTimer()
        self.wave_timer.timeout.connect(self.refresh_waveform)
        self.wave_timer.start(40)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

        # 统一 AI 推理定时器（模拟+硬件共用）：每 30s 对 90s 滑窗推理一次
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self._periodic_ai_inference)
        self.ai_timer.start(AI_INFERENCE_INTERVAL_MS)

    def _periodic_ai_inference(self):
        """定时触发 ONNX 推理，更新最近一次推理结果"""
        if not self.model_loaded:
            return
        cl, rv, probs = self.run_onnx_inference()
        if cl is not None:
            self.latest_label = cl
            self.latest_rr = rv
            self.latest_probs = probs
            # 记录本地结论，供 _apply_frame 与端侧结果做交叉验证
            self.latest_local_label = cl

    def _reset_ai_window(self):
        """清空 AI 推理窗口与最近一次推理结果（用于模式切换）。

        清空后窗口需重新累积 90 秒才能推理，期间 _is_warming_up() 为 True，
        界面显示「预热中」，不会给出跨模式的错误结论。
        """
        self.ai_data_buffer.clear()
        self.latest_probs = None
        self.latest_label = None
        self.latest_rr = 0.0
        self.latest_local_label = None
        self.latest_fw_label = None
        # 新模式重新计数，避免沿用上个模式的连续异常累计
        self._alarm_consecutive = 0
        self._last_ai_label = None

    def _window_fill_ratio(self) -> float:
        """AI 推理窗口填充比例 (0~1)，用于判断是否处于"预热中"。

        实测窗口 <60s 时诊断不可靠（急促呼吸易误判为正常），
        因此未满窗时结果仅供参考，不作为告警依据。
        """
        try:
            n = len(self.ai_data_buffer)
        except Exception:
            return 0.0
        return min(1.0, n / max(1, AI_WINDOW_POINTS))

    def _is_warming_up(self) -> bool:
        """推理窗口是否尚未填满（<90s 数据）"""
        return self._window_fill_ratio() < 1.0

    # ═══════════════════════════════════════════
    #  模拟数据生成
    # ═══════════════════════════════════════════

    def setup_simulator(self):
        self.sim_running = False
        self.sim_tick = 0
        self.sim_pattern_index = 0
        self.sim_phase = 0.0

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.generate_sim_data)

    def toggle_simulator(self):
        if self.sim_running:
            # 暂停：只停定时器，保留所有状态，记录暂停开始时间
            self.sim_timer.stop()
            self.sim_running = False
            self.pause_start = datetime.now()
            self.auto_scroll = False
            self.sim_btn.setText("▶ 继续模拟")
            self.sim_btn.setStyleSheet(
                "QPushButton { background-color:#FF6600; color:white; "
                "font-weight:bold; border-radius:4px; padding:4px; }"
            )
            self.clear_btn.setEnabled(True)
            self.sim_pattern_label.setText("(已暂停，可拖拽波形回看)")
            self.scroll_hint.setText("已暂停 | 左键拖拽平移 | 滚轮缩放 | 右键框选放大")
            self.statusBar().showMessage('模拟已暂停 — 可拖拽查看历史波形，按「继续」从当前位置恢复')
        else:
            is_resume = len(self.wave_buffer) > 0  # 缓冲区有数据 = 恢复，否则首次启动
            # 累加暂停时长，保证恢复后时间戳连续
            if self.pause_start is not None:
                self.total_pause_duration += (datetime.now() - self.pause_start).total_seconds()
                self.pause_start = None
            if not is_resume:
                # 首次启动：重置所有状态
                self.sim_tick = 0
                self.sim_pattern_index = 0
                self.sim_phase = 0.0
                self.start_time = datetime.now()
                self.total_pause_duration = 0.0
                self.pause_start = None
                self.wave_buffer.clear()
                self.wave_timestamps.clear()
                self._reset_ai_window()
                self._reset_stage3_session("模拟启动/新会话")
                self.packet_count = 0
                ai_mode = "ONNX 模型推理" if self.model_loaded else "硬编码标签"
                self.log_msg(f"═══ 模拟启动（AI模式: {ai_mode}）═══")
            # 恢复：sim_tick/sim_phase/sim_pattern_index 保持原值，从当前位置继续
            self.auto_scroll = True
            self.sim_timer.start(200)
            self.sim_running = True
            self.clear_btn.setEnabled(False)
            self.sim_btn.setText("⏸ 暂停")
            self.sim_btn.setStyleSheet(
                "QPushButton { background-color:#CC0000; color:white; "
                "font-weight:bold; border-radius:4px; padding:4px; }"
            )
            self.scroll_hint.setText("运行中：自动滚动 | 暂停后可拖拽查看历史")
            self.statusBar().showMessage("模拟运行中 — 每30秒运行一次 ONNX 模型推理")

    def clear_simulator(self):
        """清除当前波形，回到初始状态"""
        self.sim_tick = 0
        self.sim_pattern_index = 0
        self.sim_phase = 0.0
        self.start_time = datetime.now()
        self.total_pause_duration = 0.0
        self.pause_start = None
        self.wave_buffer.clear()
        self.wave_timestamps.clear()
        self._reset_ai_window()
        self._reset_stage3_session("模拟清除")
        self.packet_count = 0
        self.log.clear()
        self.clear_btn.setEnabled(False)
        self.sim_btn.setText("▶ 启动模拟")
        self.sim_btn.setStyleSheet(
            "QPushButton { background-color:#FF6600; color:white; "
            "font-weight:bold; border-radius:4px; padding:4px; }"
        )
        self.sim_pattern_label.setText("(已清除)")
        self.ai_label.setText("点击「启动模拟」查看效果")
        self.confidence_label.setText("")
        self.statusBar().showMessage("波形已清除 — 点击「启动模拟」重新开始")
        self.curve.setData([], [])  # 强制清空波形图

    @staticmethod
    def _capnogram(phase: float, pattern: dict) -> float:
        """真实呼吸波形（capnogram）— 四相模型

        相位分布（与训练数据形态一致，是模型能正确分类的关键）：
            吸气相 (0 ~ 0.10)          : 基线，CO2 接近 0
            上升段 (0.10 ~ 0.25)       : 快速上升
            呼气平台 (0.25 ~ 0.25+f3)  : 缓慢升至 EtCO2（呼气末气流停滞）
            下降段 (其余)              : 快速回落

        参数取自 pattern：rr / etco2 / base / f3
        """
        f1, f2 = 0.10, 0.15          # 吸气相、上升段占比
        f3 = pattern.get("f3", 0.60)  # 呼气平台占比
        etco2 = pattern["etco2"]
        base = pattern["base"]

        if phase < f1:
            return base
        elif phase < f1 + f2:
            return base + (etco2 - base) * ((phase - f1) / f2)
        elif phase < f1 + f2 + f3:
            frac = (phase - f1 - f2) / f3
            return base + (etco2 - base) * (0.90 + 0.10 * frac)
        else:
            frac = (phase - f1 - f2 - f3) / max(1e-6, 1.0 - f1 - f2 - f3)
            return etco2 * (1 - frac * 0.95)

    def generate_sim_data(self):
        """每 200ms 生成 5 个采样点"""
        pattern = SIM_PATTERNS[self.sim_pattern_index]

        # 模式切换
        elapsed = self.sim_tick * 0.200
        if elapsed >= pattern["duration"]:
            self.sim_pattern_index = (self.sim_pattern_index + 1) % len(SIM_PATTERNS)
            self.sim_phase = 0.0
            self.sim_tick = 0
            pattern = SIM_PATTERNS[self.sim_pattern_index]
            self.log_msg(f"──> 切换模式: {pattern['name']}")
            # 关键：清空 AI 推理缓冲，避免 90s 滑动窗口横跨模式边界。
            # 否则切换后前 90 秒窗口里大部分仍是上一个模式的波形，
            # 模型看到混合波形会给出错误结论（实测屏息模式 70% 时间判错）。
            self._reset_ai_window()
            self.log_msg("    （已重置推理窗口，约 90 秒后给出本模式诊断）")

        concentrations_ppm = []
        for j in range(5):
            breath_freq = pattern["rr"] / 60.0  # Hz
            self.sim_phase += breath_freq * 0.040
            if self.sim_phase >= 1.0:
                self.sim_phase -= 1.0

            co2 = self._capnogram(self.sim_phase, pattern)
            co2 += random.gauss(0, 0.8)
            # v2.3：帧内改为传浓度(ppm)，此处按与固件一致的换算反向生成
            concentrations_ppm.append(mmhg_to_ppm(max(0.0, co2)))

            # AI 推理缓冲由 _apply_frame 统一喂入（模拟+硬件共用），这里不再单独 append

        # ── AI 标签兜底逻辑（推理由 ai_timer 每 30s 驱动）──
        ai_class = None
        rr_val = 0.0
        # 有最近一次推理结果 → 用模型结论
        if self.model_loaded and getattr(self, 'latest_probs', None) is not None:
            ai_class = getattr(self, 'latest_label', None)
            rr_val = getattr(self, 'latest_rr', 0.0)
        # 无推理结果（模型未加载 / 90s 窗口未满）→ 用硬编码标签兜底保持 UI 有显示
        if ai_class is None:
            ai_class = pattern["truth"]
            rr_val = pattern["rr"]

        # 编码帧 → 解码 → 应用（真实模拟 STM32 → 上位机数据流）
        raw_frame = encode_frame(concentrations_ppm, ai_class, rr_val)
        frame = decode_frame(raw_frame)
        if frame:
            self._apply_frame(frame)

        self.sim_pattern_label.setText(f"模拟: {pattern['name']}")
        self.sim_tick += 1

    # ═══════════════════════════════════════════
    #  ONNX 推理 — 和 STM32 的 Extract_Features + ai_network_run 完全等价
    # ═══════════════════════════════════════════

    # ── 寻峰 / 特征子函数（与训练端 v2 features/extract_features.py 1:1 一致）──

    def _find_breath_peaks(self, arr, fs=AI_SAMPLE_RATE):
        """在 CO2 波形上找有效呼吸峰（与训练端 scipy.find_peaks 参数一致）"""
        peaks, _ = find_peaks(arr, height=12,
                              distance=int(1.0 * fs),
                              prominence=8)
        return peaks

    def _feature_stat(self, arr):
        """F1-F4：均值 / 标准差 / 峰峰值 / 偏度（三阶标准矩）"""
        x = np.asarray(arr, dtype=float)
        mean = float(np.mean(x))
        std = float(np.std(x))
        peak_to_peak = float(np.max(x) - np.min(x))
        if std > 1e-6:
            skew = float(np.mean((x - mean) ** 3) / (std ** 3))
        else:
            skew = 0.0
        return [mean, std, peak_to_peak, skew]

    def _feature_periodicity(self, peaks, fs=AI_SAMPLE_RATE):
        """F5-F6：峰间距均值/标准差（秒），峰<2 返回 [0,0]"""
        if len(peaks) < 2:
            return [0.0, 0.0]
        intervals = np.diff(peaks)
        intervals_sec = intervals / fs
        return [float(np.mean(intervals_sec)), float(np.std(intervals_sec))]

    def _feature_width(self, arr, peaks, fs=AI_SAMPLE_RATE):
        """F7：有效呼吸峰的平均宽度（秒），峰=0 返回 0"""
        if len(peaks) == 0:
            return 0.0
        x = np.asarray(arr, dtype=float)
        n = len(x)
        half_height = 7.5
        widths = []
        for p in peaks:
            peak_val = x[p]
            threshold = peak_val - half_height
            left = p
            while left > 0 and x[left] > threshold:
                left -= 1
            right = p
            while right < n - 1 and x[right] > threshold:
                right += 1
            widths.append((right - left) / fs)
        return float(np.mean(widths)) if widths else 0.0

    def _feature_plateau(self, arr, peaks, tolerance=3.0):
        """F8：呼气平台占比（距最大峰 3.0 以内点占比），峰=0 或 max≤0 返回 0"""
        if len(peaks) == 0:
            return 0.0
        x = np.asarray(arr, dtype=float)
        max_val = np.max(x)
        if max_val <= 0:
            return 0.0
        plateau_points = np.sum(x >= (max_val - tolerance))
        return float(plateau_points / len(x))

    def _feature_peak_count_amp(self, arr, peaks):
        """F9-F10：呼吸峰数 / 平均峰幅（峰高减基线）"""
        x = np.asarray(arr, dtype=float)
        n_peaks = len(peaks)
        if n_peaks > 0:
            baseline = np.min(x)
            amplitudes = x[peaks] - baseline
            mean_amp = float(np.mean(amplitudes))
        else:
            mean_amp = 0.0
        return [float(n_peaks), mean_amp]

    def _feature_wave_rr(self, peaks):
        """F11：波形自算呼吸率 bpm = n_peaks × 60/30 = n_peaks × 2
        ★ 必须照抄训练端公式（保持 30s 语义），不能"修正"成 90s —— 模型按此值训练
        """
        n_peaks = len(peaks)
        if n_peaks == 0:
            return 0.0
        return float(n_peaks * 60.0 / 30.0)

    def _feature_rhythm_confidence(self, peaks):
        """F12：节律置信度 max(0, 1 - 峰间距变异系数)，峰<3 返回 0"""
        if len(peaks) < 3:
            return 0.0
        intervals = np.diff(peaks)
        mean_iv = np.mean(intervals)
        if mean_iv < 1e-6:
            return 0.0
        std_iv = np.std(intervals)
        cv = std_iv / mean_iv
        return float(max(0.0, 1.0 - cv))

    def extract_features(self, data, fs=AI_SAMPLE_RATE):
        """
        特征提取 — 与训练端 v2 features/extract_features.py 完全一致
        输入: 2250 个 CO2 mmHg 值 (90 秒 × 25Hz)
        输出: (features_12, n_peaks)
              features_12 顺序 = [F1..F12]（均值/标准差/峰峰值/偏度/峰间距均值/峰间距标准差/
                                        峰宽/平台占比/峰数/峰幅/波形自算RR/节律置信度）
              n_peaks = 有效呼吸峰数（用于显示真实呼吸率）
        """
        if len(data) == 0:
            return [0.0] * 12, 0

        x = np.asarray(data, dtype=float)

        # 寻峰
        peaks = self._find_breath_peaks(x, fs)

        # F1-F4 基础统计
        feat_stat = self._feature_stat(x)
        # F5-F6 周期结构
        feat_period = self._feature_periodicity(peaks, fs)
        # F7 峰宽
        feat_width = self._feature_width(x, peaks, fs)
        # F8 平台占比
        feat_plateau = self._feature_plateau(x, peaks)
        # F9-F10 峰数/峰幅
        feat_count_amp = self._feature_peak_count_amp(x, peaks)
        # F11-F12 波形自算 RR / 节律置信度
        feat_wave_rr = self._feature_wave_rr(peaks)
        feat_rhythm = self._feature_rhythm_confidence(peaks)

        features = (feat_stat + feat_period + [feat_width, feat_plateau]
                    + feat_count_amp + [feat_wave_rr, feat_rhythm])
        return [float(v) for v in features], int(len(peaks))

    def run_onnx_inference(self):
        """用 v2 ONNX 模型推理当前缓冲区中的 90 秒数据（模拟+硬件共用）

        返回:
            (label, display_rr, prob_dict) 或 (None, None, None)
            - label: 4 类标签（0/1/2/3）
            - display_rr: 真实呼吸率 bpm（n_peaks × 60 / 90，仅用于显示，区别于 F11）
            - prob_dict: ZipMap 概率字典（键 0/1/2/3），供置信度读取
        """
        if not self.model_loaded:
            return None, None, None

        try:
            data = list(self.ai_data_buffer)[-AI_WINDOW_POINTS:]  # 取最近 2250 点 (90s)
            if len(data) < AI_MIN_POINTS:
                return None, None, None

            features, n_peaks = self.extract_features(data)
            # ONNX 输入 shape: [1, 12] = [batch, features]（与 v2 训练时 12 维特征一致）
            input_data = np.array([features], dtype=np.float32)

            outputs = self.ort_session.run(
                self.ort_output_names,
                {self.ort_input_name: input_data}
            )

            # output[0] = output_label (int64), output[1] = output_probability (ZipMap dict)
            label = int(outputs[0].flatten()[0])

            # ZipMap 输出为 dict，键 = classlabels [0,1,2,3] → 保留原始 dict
            prob_dicts = outputs[1] if len(outputs) > 1 else []
            if prob_dicts and isinstance(prob_dicts, list) and len(prob_dicts) > 0:
                prob_dict = dict(prob_dicts[0])
            else:
                prob_dict = None

            # 从概率字典取置信度（键缺失时视为高置信度）
            if prob_dict:
                confidence = float(prob_dict.get(label, 1.0))
            else:
                confidence = 1.0

            # 真实呼吸率（仅显示用）：峰数 × 60 / 90s
            display_rr = (n_peaks * 60.0 / AI_WINDOW_SECONDS) if n_peaks else 0.0

            # 日志（12 维特征）
            self.log_msg(
                f"[ONNX推理] 特征=(mean={features[0]:.1f}, std={features[1]:.1f}, "
                f"ptp={features[2]:.1f}, skew={features[3]:.2f}, "
                f"pkint_mean={features[4]:.2f}, pkint_std={features[5]:.2f}, "
                f"width={features[6]:.2f}, plateau={features[7]:.2f}, "
                f"npeak={features[8]:.0f}, amp={features[9]:.1f}, "
                f"wave_rr={features[10]:.1f}, rhythm={features[11]:.2f}) "
                f"→ 结果={LABEL_MAP.get(label, '?')}"
                f"({label}) 置信度={confidence:.0%} RR≈{display_rr:.1f}"
            )

            return label, display_rr, prob_dict

        except Exception as e:
            self.log_msg(f"[ERROR] ONNX 推理失败: {e}")
            return None, None, None

    # ═══════════════════════════════════════════
    #  定时刷新
    # ═══════════════════════════════════════════

    def refresh_waveform(self):
        if len(self.wave_buffer) == 0:
            return
        buf = list(self.wave_buffer)
        t = list(self.wave_timestamps)
        if len(t) == len(buf) and len(t) > 0:
            self.curve.setData(t, buf)
            # 仅在自动滚动模式（运行中）强制 X 轴跟随最新数据
            if self.auto_scroll:
                last_t = t[-1]
                self.waveform.setXRange(max(0, last_t - 20), last_t + 0.5, padding=0)
            # 暂停时不做 setXRange，用户可自由拖拽查看任意历史区域

        if len(buf) > 10:
            peaks = []
            for i in range(1, len(buf) - 1):
                if buf[i] > buf[i - 1] and buf[i] > buf[i + 1] and buf[i] > 10:
                    peaks.append(buf[i])
            if peaks and self.display_mode != DISPLAY_MODE_RELATIVE:
                etco2 = max(peaks[-5:]) if len(peaks) >= 5 else max(peaks)
                self._update_card("EtCO₂（未标定）", f"{etco2:.1f}", "#FF9933")

    def update_stats(self):
        self.packet_count_label.setText(f"接收包: {self.packet_count}")
        # 历史标签可见时自动刷新（限频每5秒，避免频繁重建表格）
        if hasattr(self, '_history_tabs'):
            idx = self._history_tabs.currentIndex()
            if idx == 0 or idx == 1:
                self._auto_refresh_ticks = getattr(self, '_auto_refresh_ticks', 0) + 1
                if self._auto_refresh_ticks % 5 == 0:
                    try:
                        if idx == 0:
                            self._refresh_history()
                        else:
                            self._refresh_alarm_history()
                    except Exception:
                        pass

    # ═══════════════════════════════════════════
    #  辅助方法
    # ═══════════════════════════════════════════

    def _update_card(self, title, value, color):
        card = self.findChild(QLabel, f"val_{title}")
        if card:
            card.setText(value)
            card.setStyleSheet(f"color:{color}; font-size:17px; font-weight:bold;")

    # ═══════════════════════════════════════════
    #  业务逻辑：告警通知 / 趋势预警 / 历史查看 / 每日摘要
    # ═══════════════════════════════════════════

    def _setup_tray_icon(self):
        """创建系统托盘图标，支持右键菜单和告警弹窗"""
        self._tray = QSystemTrayIcon(self)
        self._tray.setToolTip("CO2 呼吸波形监控")
        # 绘制一个简单的绿色圆形图标
        from PySide6.QtGui import QPixmap, QPainter, QColor, QBrush
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#00CC66")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        painter.end()
        self._tray.setIcon(QIcon(pm))
        tray_menu = QMenu()
        tray_menu.addAction("显示窗口", self.show)
        tray_menu.addAction("退出", self.close)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(
            lambda reason: self.show() if reason == QSystemTrayIcon.DoubleClick else None)
        self._tray.show()

    def _trigger_alert(self, alarm_type: str, ai_class: int, co2: float, rr: float):
        """告警通知：状态栏闪烁 + 托盘弹窗 + 蜂鸣（弹窗/蜂鸣限频）

        告警检测逐帧运行，若不加限制，持续异常时会每 0.4 秒弹一次窗、
        蜂鸣一次，严重刷屏。这里对"弹窗 + 蜂鸣"做冷却：同一冷却期内
        只提示一次；状态栏闪烁与日志/DB 记录不受影响，异常信息不丢失。
        """
        now = datetime.now()

        # 1. 状态栏红底闪烁（每次都确保在闪，不消耗冷却）
        if not self._alert_flashing:
            self._alert_flashing = True
            self._alert_flash_timer.start(500)  # 每500ms切换

        # 2. 托盘弹窗 + 蜂鸣：受冷却限制
        should_notify = (
            self._last_alarm_time is None
            or (now - self._last_alarm_time).total_seconds() >= ALARM_NOTIFY_COOLDOWN_SEC
        )
        if should_notify:
            self._last_alarm_time = now
            msg = f"CO2={co2:.1f}mmHg  RR={rr:.1f}次/分"
            self._tray.showMessage(f"⚠ 呼吸告警: {alarm_type}", msg,
                                   QSystemTrayIcon.Warning, 5000)
            if self.config.get("alarm.sound_enabled", False):
                QApplication.beep()

        # 3. 写入日志（不消耗冷却，异常全程留痕）
        self.log_msg(f"⚠ 告警触发: {alarm_type} (CO2={co2:.1f} RR={rr:.1f})",
                     level="warn", source="alarm")

    def _flash_statusbar(self):
        """告警闪烁：交替红/深灰背景"""
        if not self._alert_flashing:
            self._alert_flash_timer.stop()
            self.statusBar().setStyleSheet("")
            return
        current = self.statusBar().styleSheet()
        if "background-color: #CC0000" in current:
            self.statusBar().setStyleSheet("background-color: #2D2D2D;")
        else:
            self.statusBar().setStyleSheet("background-color: #CC0000; color: white;")

    def _stop_alert(self):
        """停止告警闪烁（用户点击状态栏或托盘时调用）"""
        self._alert_flashing = False
        self._alert_flash_timer.stop()
        self.statusBar().setStyleSheet("")

    def _analyze_trends(self):
        """趋势预警：分析最近数据，提前发现呼吸恶化趋势"""
        rows = self.db.query_recent_device_data(limit=10)
        if len(rows) < 5:
            return  # 数据不够，跳过

        rr_values = [r['rr'] for r in rows if r['rr'] is not None]
        etco2_values = [r['etco2'] for r in rows if r['etco2'] is not None]

        warnings = []

        # ── 呼吸率趋势检测（简单线性回归）──
        if len(rr_values) >= 5:
            n = len(rr_values)
            x_mean = (n - 1) / 2.0
            y_mean = sum(rr_values) / n
            num = sum((i - x_mean) * (rr_values[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            if den > 0:
                slope = num / den
                if abs(slope) > 2.0:  # 每采样周期变化 > 2 次/分
                    direction = "上升" if slope > 0 else "下降"
                    warnings.append(f"呼吸率持续{direction} (斜率={slope:+.1f}/周期)")

        # ── EtCO2 基线偏离检测 ──
        if len(etco2_values) >= 5:
            abnormal = [e for e in etco2_values if e < 30 or e > 50]
            if len(abnormal) >= 5:
                avg_dev = sum(abnormal) / len(abnormal)
                warnings.append(f"EtCO2基线偏离 (均值={avg_dev:.1f}mmHg)")

        # 触发趋势预警
        for w in warnings:
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            self.db.insert_alarm(ts=ts, alarm_type=f"趋势预警: {w}",
                                 co2_value=rows[0].get('co2'), rr_value=rows[0].get('rr'))
            self.log_msg(f"[趋势预警] {w}", level="warn", source="trend")

    def _setup_history_tab(self):
        """在波形/日志区域增加历史数据查看标签页"""
        # 找到主布局中的 splitter，替换为 tab widget
        self._history_tabs = QTabWidget()
        self._history_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #555; }")

        # ── 页面1：历史表格 ──
        hist_page = QWidget()
        hv = QVBoxLayout(hist_page)
        hv.setContentsMargins(4, 4, 4, 4)

        # 筛选栏
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("开始:"))
        self._hist_start = QDateEdit()
        self._hist_start.setCalendarPopup(True)
        self._hist_start.setDate(datetime.now().date())
        filter_row.addWidget(self._hist_start)
        filter_row.addWidget(QLabel("结束:"))
        self._hist_end = QDateEdit()
        self._hist_end.setCalendarPopup(True)
        self._hist_end.setDate(datetime.now().date())
        filter_row.addWidget(self._hist_end)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_history)
        filter_row.addWidget(refresh_btn)

        export_btn = QPushButton("导出CSV")
        export_btn.clicked.connect(self._export_csv)
        filter_row.addWidget(export_btn)

        excel_btn = QPushButton("导出Excel")
        excel_btn.clicked.connect(self._export_excel)
        filter_row.addWidget(excel_btn)

        report_btn = QPushButton("生成报告")
        report_btn.setStyleSheet(
            "QPushButton { background-color:#00AA00; color:white; "
            "font-weight:bold; border-radius:4px; padding:4px 10px; }")
        report_btn.clicked.connect(self._generate_report)
        filter_row.addWidget(report_btn)

        self._hist_stats = QLabel("")
        filter_row.addWidget(self._hist_stats)
        filter_row.addStretch()
        hv.addLayout(filter_row)

        # 表格
        self._hist_table = QTableWidget()
        self._hist_table.setColumnCount(6)
        self._hist_table.setHorizontalHeaderLabels(
            ["时间", "相对 CO₂", "呼吸率", "EtCO₂（未标定）",
             "旧固件分类（仅参考）", "来源"])
        self._hist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._hist_table.setAlternatingRowColors(True)
        self._hist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        hv.addWidget(self._hist_table)
        self._history_tabs.addTab(hist_page, "历史数据")

        # ── 页面2：告警记录 ──
        alarm_page = QWidget()
        av = QVBoxLayout(alarm_page)
        av.setContentsMargins(4, 4, 4, 4)
        alarm_filter = QHBoxLayout()
        alarm_filter.addWidget(QLabel("告警记录"))
        alarm_refresh = QPushButton("刷新")
        alarm_refresh.clicked.connect(self._refresh_alarm_history)
        alarm_filter.addWidget(alarm_refresh)
        alarm_filter.addStretch()
        av.addLayout(alarm_filter)
        self._alarm_table = QTableWidget()
        self._alarm_table.setColumnCount(5)
        self._alarm_table.setHorizontalHeaderLabels(
            ["时间", "类型", "CO2", "呼吸率", "置信度"])
        self._alarm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._alarm_table.setAlternatingRowColors(True)
        self._alarm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        av.addWidget(self._alarm_table)
        self._history_tabs.addTab(alarm_page, "告警记录")

        # ── 页面3：趋势回放（生产 DB 趋势，仅查看）──
        self._setup_playback_tab()
        # ── 页面4：Stage 3 隔离回放工作台 ──
        self._setup_stage3_replay_tab()

        # 切到历史标签页时自动刷新数据
        self._history_tabs.currentChanged.connect(self._on_history_tab_changed)

    def _on_history_tab_changed(self, index: int):
        """切换到历史/告警/回放标签页时自动刷新"""
        try:
            if index == 0:
                self._refresh_history()
            elif index == 1:
                self._refresh_alarm_history()
            elif index == 2 and not self._playback_rows:
                # 回放 tab 首次进入且未加载数据时，自动加载最近一天
                self._load_playback_data()
        except Exception:
            pass

    def _on_main_tab_changed(self, index: int):
        """切换顶层 tab：进入历史页时刷新子页内容"""
        try:
            if index == 1:
                self._refresh_history()
                self._refresh_alarm_history()
        except Exception:
            pass

    # ═══════════════════════════════════════════
    #  趋势回放
    # ═══════════════════════════════════════════

    def _setup_playback_tab(self):
        """构建趋势回放页面：控制栏 + 趋势曲线图 + AI标签色条"""
        self._playback_page = QWidget()
        pv = QVBoxLayout(self._playback_page)
        pv.setContentsMargins(4, 4, 4, 4)

        # ── 控制栏 ──
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("开始:"))
        self._pb_start = QDateEdit()
        self._pb_start.setCalendarPopup(True)
        self._pb_start.setDate(datetime.now().date())
        ctrl.addWidget(self._pb_start)
        ctrl.addWidget(QLabel("结束:"))
        self._pb_end = QDateEdit()
        self._pb_end.setCalendarPopup(True)
        self._pb_end.setDate(datetime.now().date())
        ctrl.addWidget(self._pb_end)

        load_btn = QPushButton("加载数据")
        load_btn.clicked.connect(self._load_playback_data)
        ctrl.addWidget(load_btn)

        self._pb_play_btn = QPushButton("▶ 播放")
        self._pb_play_btn.clicked.connect(self._start_pause_playback)
        ctrl.addWidget(self._pb_play_btn)

        ctrl.addWidget(QLabel("倍速:"))
        self._pb_speed = QComboBox()
        self._pb_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self._pb_speed.setCurrentText("1x")
        self._pb_speed.currentTextChanged.connect(self._apply_playback_speed)
        ctrl.addWidget(self._pb_speed)

        self._pb_elapsed = QLabel("")
        self._pb_elapsed.setStyleSheet("color:#888888;")
        ctrl.addWidget(self._pb_elapsed)
        ctrl.addStretch()
        pv.addLayout(ctrl)

        # ── 当前读数面板（播放时大号显示）──
        readout = QHBoxLayout()
        readout.addWidget(QLabel("当前:"))
        self._pb_readout = QLabel("加载数据后开始回放")
        self._pb_readout.setStyleSheet(
            "color:#00CC66; font-size:18px; font-weight:bold; padding:4px 8px;")
        readout.addWidget(self._pb_readout)
        readout.addStretch()
        pv.addLayout(readout)

        # ── 进度条 ──
        self._pb_slider = QSlider(Qt.Horizontal)
        self._pb_slider.setRange(0, 0)
        self._pb_slider.valueChanged.connect(self._playback_slider_moved)
        pv.addWidget(self._pb_slider)

        # ── 曲线图 + 标签条 放进垂直 QSplitter（可拖拽调整高度）──
        self._pb_splitter = QSplitter(Qt.Vertical)
        self._pb_splitter.setChildrenCollapsible(False)

        # ── 呼吸率图（独立图，左轴=次/分）──
        self._pb_rr_plot = pg.PlotWidget()
        self._pb_rr_plot.setBackground('#1A1A1A')
        self._pb_rr_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pb_rr_plot.setLabel('left', '呼吸率', units='次/分')
        self._pb_rr_plot.setLabel('bottom', '时间', units='秒')
        self._pb_rr_curve = self._pb_rr_plot.plot(
            pen=pg.mkPen(color=(0, 153, 255), width=2), name='呼吸率')
        # 呼吸率正常范围带 12-20
        self._pb_rr_band = pg.LinearRegionItem(
            [12, 20], brush=pg.mkBrush(0, 153, 255, 22),
            movable=False, pen=None)
        self._pb_rr_plot.addItem(self._pb_rr_band)
        # 呼吸率图播放游标
        self._pb_cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255), width=1, style=Qt.DashLine))
        self._pb_rr_plot.addItem(self._pb_cursor)

        # ── EtCO2 图（独立图，左轴=mmHg）──
        self._pb_etco2_plot = pg.PlotWidget()
        self._pb_etco2_plot.setBackground('#1A1A1A')
        self._pb_etco2_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pb_etco2_plot.setLabel('left', 'EtCO₂（未标定）')
        self._pb_etco2_curve = self._pb_etco2_plot.plot(
            pen=pg.mkPen(color=(255, 153, 0), width=2), name='EtCO2')
        # EtCO2 正常范围带 35-45
        self._pb_etco2_band = pg.LinearRegionItem(
            [35, 45], brush=pg.mkBrush(255, 153, 0, 20),
            movable=False, pen=None)
        self._pb_etco2_plot.addItem(self._pb_etco2_band)
        self._pb_etco2_band.setVisible(False)
        try:
            self._pb_etco2_plot.getAxis('left').setStyle(showValues=False)
        except Exception:
            pass
        # EtCO2 图播放游标
        self._pb_etco2_cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255), width=1, style=Qt.DashLine))
        self._pb_etco2_plot.addItem(self._pb_etco2_cursor)

        # 两图共享 X 轴（缩放/平移联动）
        self._pb_etco2_plot.setXLink(self._pb_rr_plot)

        # AI 标签色条（下方）
        label_bar_widget = QWidget()
        lbv = QVBoxLayout(label_bar_widget)
        lbv.setContentsMargins(0, 0, 0, 0)
        self._pb_label_bar = pg.PlotWidget()
        self._pb_label_bar.setBackground('#1A1A1A')
        self._pb_label_bar.hideAxis('left')
        self._pb_label_bar.setYRange(0, 1)
        self._pb_label_cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255), width=1, style=Qt.DashLine))
        self._pb_label_bar.addItem(self._pb_label_cursor)
        lbv.addWidget(self._pb_label_bar)
        # 图例行
        legend_row = QHBoxLayout()
        legend_row.addStretch()
        legend_items = [("正常", "#00CC66"), ("急促", "#FF9933"),
                        ("屏息", "#FF4444"), ("异常", "#FFDD44")]
        for text, color in legend_items:
            swatch = QLabel("■■")
            swatch.setStyleSheet(f"color:{color}; font-size:12px;")
            legend_row.addWidget(swatch)
            legend_row.addWidget(QLabel(text))
        legend_row.addStretch()
        lbv.addLayout(legend_row)

        self._pb_splitter.addWidget(self._pb_rr_plot)
        self._pb_splitter.addWidget(self._pb_etco2_plot)
        self._pb_splitter.addWidget(label_bar_widget)
        self._pb_splitter.setSizes([300, 300, 90])
        pv.addWidget(self._pb_splitter, stretch=3)

        self._history_tabs.addTab(self._playback_page, "趋势回放")

    # ============================================================
    #  Stage 3 isolated replay workbench (no production DB/notifications)
    # ============================================================

    def _setup_stage3_replay_tab(self):
        page = QWidget()
        pv = QVBoxLayout(page)
        pv.setContentsMargins(4, 4, 4, 4)

        banner = QLabel("REPLAY · 隔离回放（开发集/合成数据） · 不写生产库 · 不触发生产通知")
        banner.setStyleSheet(
            "color:#FFFFFF; background-color:#7A3E00; font-weight:bold; "
            "border-radius:4px; padding:6px;")
        pv.addWidget(banner)

        ctrl = QHBoxLayout()
        load_btn = QPushButton("加载回放包")
        load_btn.clicked.connect(lambda: self._load_replay_pack_path(None))
        ctrl.addWidget(load_btn)
        self._rp_play_btn = QPushButton("▶ 播放")
        self._rp_play_btn.clicked.connect(self._replay_play_pause)
        ctrl.addWidget(self._rp_play_btn)
        restart_btn = QPushButton("⟲ 重新开始")
        restart_btn.clicked.connect(self._replay_restart)
        ctrl.addWidget(restart_btn)
        ctrl.addWidget(QLabel("引擎:"))
        self._rp_engine = QComboBox()
        self._rp_engine.addItems(["Stage3（默认）", "Stage4实验（四尺度）"])
        self._rp_engine.setToolTip(
            "Stage4 四尺度融合为外部审查实验项；未通过收益判据时不得作为默认路径")
        ctrl.addWidget(self._rp_engine)
        ctrl.addWidget(QLabel("倍速:"))
        self._rp_speed = QComboBox()
        self._rp_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self._rp_speed.setCurrentText("1x")
        self._rp_speed.currentTextChanged.connect(
            lambda _: self._replay_update_view())
        ctrl.addWidget(self._rp_speed)
        ctrl.addStretch()
        pv.addLayout(ctrl)

        self._rp_source = QLabel("未加载 · Replay")
        self._rp_source.setStyleSheet("color:#FFCC66; font-weight:bold;")
        pv.addWidget(self._rp_source)
        self._rp_state = QLabel("等待加载回放包")
        self._rp_state.setWordWrap(True)
        self._rp_state.setStyleSheet(
            "color:#CCCCCC; background-color:#202020; border-radius:5px; padding:6px;")
        pv.addWidget(self._rp_state)

        self._rp_slider = QSlider(Qt.Horizontal)
        self._rp_slider.setRange(0, 0)
        self._rp_slider.valueChanged.connect(self._replay_slider_moved)
        pv.addWidget(self._rp_slider)

        self._rp_plot = pg.PlotWidget()
        self._rp_plot.setBackground('#1A1A1A')
        self._rp_plot.setLabel('left', '相对 CO₂')
        self._rp_plot.setLabel('bottom', '信号时间', units='s')
        try:
            self._rp_plot.getAxis('left').setStyle(showValues=False)
        except Exception:
            pass
        self._rp_curve = self._rp_plot.plot(
            pen=pg.mkPen(color=(255, 204, 102), width=1))
        self._rp_cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255), width=1, style=Qt.DashLine))
        self._rp_plot.addItem(self._rp_cursor)
        pv.addWidget(self._rp_plot, stretch=3)
        self._history_tabs.addTab(page, "隔离回放")

    def _default_replay_pack(self):
        return Path(__file__).resolve().parent / "replay_data" / "dev_demo.json"

    def _load_replay_pack_path(self, path):
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            default = str(self._default_replay_pack())
            path, _ = QFileDialog.getOpenFileName(
                self, "加载 Stage3 隔离回放包", default,
                "Replay pack (*.json)")
            if not path:
                return
        try:
            pack = ReplayPack.load(path)
            engine_name = (self._rp_engine.currentText()
                           if hasattr(self, "_rp_engine") else "Stage3（默认）")
            if engine_name.startswith("Stage4"):
                runtime = Stage4Runtime()
                engine_label = "Stage4实验（四尺度）"
            else:
                runtime = Stage3Runtime()
                engine_label = "Stage3（默认）"
            if not runtime.available:
                raise RuntimeError(runtime.error or "事件引擎不可用")
            session = ReplaySession(pack, runtime)
            self._replay_pack = pack
            self._replay_session = session
            self._replay_runtime = runtime
            self._replay_engine_name = engine_label
            self._replay_timer.stop()
            self._rp_play_btn.setText("▶ 播放")
            self._rp_source.setText(
                f"REPLAY · {engine_label} · source={pack.source} · "
                f"{session.stitched_label} · {pack.duration_sec:.1f}s · "
                f"不写生产库")
            t = [i / pack.sample_rate_hz for i in range(len(pack.samples_mmhg))]
            self._rp_curve.setData(t, pack.samples_mmhg)
            self._rp_plot.setXRange(0, max(pack.duration_sec, 1))
            self._rp_slider.blockSignals(True)
            self._rp_slider.setRange(0, max(len(pack.samples_mmhg) - 1, 0))
            self._rp_slider.setValue(0)
            self._rp_slider.blockSignals(False)
            self._replay_update_view()
            self.log_msg(f"[Replay] 已加载 {path}", source="replay")
        except Exception as exc:
            self._replay_pack = None
            self._replay_session = None
            self._rp_source.setText("Replay 加载失败")
            self._rp_state.setText(str(exc))
            self.log_msg(f"[Replay] 加载失败: {exc}", level="error",
                         source="replay")

    def _replay_speed_factor(self) -> float:
        try:
            return float(self._rp_speed.currentText().rstrip("x"))
        except Exception:
            return 1.0

    def _replay_play_pause(self):
        if self._replay_session is None:
            default = self._default_replay_pack()
            if default.exists():
                self._load_replay_pack_path(str(default))
            if self._replay_session is None:
                self._rp_state.setText("请先加载回放包")
                return
        sess = self._replay_session
        if sess.playing:
            sess.pause()
            self._replay_timer.stop()
            self._rp_play_btn.setText("▶ 播放")
        else:
            sess.play()
            self._replay_timer.start(100)
            self._rp_play_btn.setText("⏸ 暂停")
        self._replay_update_view()

    def _replay_restart(self):
        if self._replay_session is None:
            self._rp_state.setText("请先加载回放包")
            return
        self._replay_timer.stop()
        self._replay_session.restart()
        self._rp_play_btn.setText("▶ 播放")
        self._rp_slider.blockSignals(True)
        self._rp_slider.setValue(0)
        self._rp_slider.blockSignals(False)
        self._replay_update_view()

    def _replay_slider_moved(self, value):
        if self._replay_session is None or self._replay_pack is None:
            return
        self._replay_timer.stop()
        self._replay_session.pause()
        self._rp_play_btn.setText("▶ 播放")
        t = value / self._replay_pack.sample_rate_hz
        # seek rebuilds history from the start; no future evidence is kept.
        self._replay_session.seek(t)
        self._replay_update_view()

    def _replay_tick(self):
        if self._replay_session is None:
            self._replay_timer.stop()
            return
        self._replay_session.advance(0.1, self._replay_speed_factor())
        if not self._replay_session.playing:
            self._replay_timer.stop()
            self._rp_play_btn.setText("▶ 播放")
        self._rp_slider.blockSignals(True)
        self._rp_slider.setValue(self._replay_session.position)
        self._rp_slider.blockSignals(False)
        self._replay_update_view()

    def _replay_update_view(self):
        sess = self._replay_session
        if sess is None:
            return
        t = sess.signal_time
        self._rp_cursor.setValue(t)
        states = sess.runtime.states() if sess.runtime is not None else {}
        if states:
            names = [s for s in ("U15", "U30", "U60", "U90", "FUSION")
                     if s in states]
            parts = [f"{s}:{states[s]['state']}" for s in names]
            self._rp_state.setText(
                f"Replay t={t:.1f}s | " + " | ".join(parts) +
                " | 拼接=" + ("是" if sess.pack.stitched else "否") +
                " | 不进入生产告警")
        else:
            self._rp_state.setText(f"Replay t={t:.1f}s · 等待数据")

    def _load_playback_data(self):
        """从 DB 加载时间段数据并绘制趋势曲线"""
        start = self._pb_start.date().toString("yyyy-MM-dd") + "T00:00:00"
        end = self._pb_end.date().toString("yyyy-MM-dd") + "T23:59:59"
        rows = self.db.query_device_data(start_ts=start, end_ts=end, limit=20000)
        if not rows:
            self.log_msg("[回放] 所选时段无数据", level="warn", source="ui")
            self._playback_rows = []
            self._pb_play_btn.setText("▶ 播放")
            self._pb_playing = False
            return

        # 转正序（DB 按时间倒序返回）
        rows.reverse()
        self._playback_rows = rows
        self._playback_pos = 0
        self._stop_playback_timer()

        # 相对时间轴（秒）
        t0 = rows[0]['timestamp']
        try:
            t0_dt = datetime.strptime(t0[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            t0_dt = datetime.now()
        times = []
        rr_vals = []
        etco2_vals = []
        co2_vals = []
        for i, r in enumerate(rows):
            try:
                ts_dt = datetime.strptime(r['timestamp'][:19], "%Y-%m-%dT%H:%M:%S")
                t = (ts_dt - t0_dt).total_seconds()
            except Exception:
                t = i
            times.append(t)
            # None 用前值填充，保持曲线连续
            rr_vals.append(r['rr'] if r['rr'] is not None else
                           (rr_vals[-1] if rr_vals else 0))
            etco2_vals.append(r['etco2'] if r['etco2'] is not None else
                              (etco2_vals[-1] if etco2_vals else 0))

        # 绘制两图曲线（各自独立 Y 轴）
        self._pb_rr_curve.setData(times, rr_vals)
        if self.display_mode == DISPLAY_MODE_RELATIVE:
            self._pb_etco2_curve.setData([], [])  # 相对模式隐藏绝对 EtCO2
        else:
            self._pb_etco2_curve.setData(times, etco2_vals)
        # 各自 Y 轴范围：呼吸率 0-40，EtCO2 0-60
        self._pb_rr_plot.setYRange(0, 40)
        self._pb_etco2_plot.setYRange(0, 60)

        # AI 标签色条：每段一个色块
        self._pb_label_bar.clear()
        self._pb_label_cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255), width=1, style=Qt.DashLine))
        self._pb_label_bar.addItem(self._pb_label_cursor)
        bar_colors = {0: '#00CC66', 1: '#FF9933', 2: '#FF4444', 3: '#FFDD44'}
        max_t = times[-1] if times else 1
        # 合并相邻同标签段，减少色块数量
        segments = []  # (start_t, end_t, label)
        for i, r in enumerate(rows):
            label = r['ai_label'] if r['ai_label'] is not None else None
            if segments and segments[-1][2] == label:
                segments[-1][1] = times[i]
            else:
                segments.append([times[i], times[i], label])
        # 修正最后一段结束时间
        from PySide6.QtGui import QColor
        for seg in segments:
            if seg[2] is None:
                color = QColor('#555555')
            else:
                color = QColor(bar_colors.get(seg[2], '#555555'))
            color.setAlpha(160)
            seg[1] = max(seg[1], seg[0] + 0.5)
            band = pg.LinearRegionItem(
                [seg[0], seg[1]], brush=color,
                movable=False, pen=None)
            band.setZValue(1)
            self._pb_label_bar.addItem(band)

        # 进度条
        n = len(times)
        self._pb_slider.setRange(0, max(n - 1, 0))
        self._pb_slider.setValue(0)
        self._pb_elapsed.setText(
            f"{len(rows)} 条 | {self._pb_start.date().toString('MM-dd')} ~ "
            f"{self._pb_end.date().toString('MM-dd')}")

        # 游标回到起点
        self._pb_cursor.setValue(times[0])
        self._pb_etco2_cursor.setValue(times[0])
        self._pb_label_cursor.setValue(times[0])
        self._update_playback_cursor(times[0], rows[0])
        self.log_msg(f"[回放] 已加载 {len(rows)} 条数据", source="ui")

    def _start_pause_playback(self):
        """播放/暂停切换"""
        if not self._playback_rows:
            self.log_msg("[回放] 请先点击「加载数据」", level="warn", source="ui")
            return
        if self._playback_playing:
            self._stop_playback_timer()
            self._pb_play_btn.setText("▶ 播放")
            self._playback_playing = False
        else:
            if self._playback_pos >= len(self._playback_rows) - 1:
                self._playback_pos = 0  # 播完重播
                self._pb_slider.setValue(0)
            self._playback_playing = True
            self._pb_play_btn.setText("⏸ 暂停")
            self._apply_playback_speed()

    def _apply_playback_speed(self):
        """根据倍速设置定时器间隔"""
        speed_text = self._pb_speed.currentText()
        speed = float(speed_text.rstrip('x'))
        interval = max(50, int(1000 / speed))
        if self._playback_playing:
            self._playback_timer.start(interval)

    def _playback_tick(self):
        """定时器回调：推进进度"""
        if not self._playback_rows:
            self._stop_playback_timer()
            return
        if self._playback_pos >= len(self._playback_rows) - 1:
            # 播到末尾自动暂停
            self._stop_playback_timer()
            self._pb_play_btn.setText("▶ 播放")
            self._playback_playing = False
            return
        self._playback_pos += 1
        self._pb_slider.setValue(self._playback_pos)
        row = self._playback_rows[self._playback_pos]
        # 更新游标
        t0 = self._playback_rows[0]['timestamp']
        try:
            t0_dt = datetime.strptime(t0[:19], "%Y-%m-%dT%H:%M:%S")
            ts_dt = datetime.strptime(row['timestamp'][:19], "%Y-%m-%dT%H:%M:%S")
            t = (ts_dt - t0_dt).total_seconds()
        except Exception:
            t = self._playback_pos
        self._update_playback_cursor(t, row)

    def _playback_slider_moved(self, value):
        """手动拖进度条"""
        if not self._playback_rows or value >= len(self._playback_rows):
            return
        self._playback_pos = value
        row = self._playback_rows[value]
        t0 = self._playback_rows[0]['timestamp']
        try:
            t0_dt = datetime.strptime(t0[:19], "%Y-%m-%dT%H:%M:%S")
            ts_dt = datetime.strptime(row['timestamp'][:19], "%Y-%m-%dT%H:%M:%S")
            t = (ts_dt - t0_dt).total_seconds()
        except Exception:
            t = value
        self._update_playback_cursor(t, row)

    def _update_playback_cursor(self, t: float, row: dict):
        """更新游标线、当前标签高亮和大号读数面板"""
        self._pb_cursor.setValue(t)
        self._pb_etco2_cursor.setValue(t)
        self._pb_label_cursor.setValue(t)
        label = row.get('ai_label')
        name = LABEL_MAP.get(label, "无诊断") if label is not None else "无诊断"
        color = LABEL_COLORS.get(label, "#AAAAAA") if label is not None else "#AAAAAA"
        conf = (row.get('confidence') or 0)
        rr = row.get('rr')
        etco2 = row.get('etco2')
        rr_str = f"{rr:.1f}" if rr is not None else "--"
        etco2_str = f"{etco2:.1f}" if etco2 is not None else "--"
        # 大号读数面板：相对模式不显示绝对 EtCO2/mmHg
        if self.display_mode == DISPLAY_MODE_RELATIVE:
            self._pb_readout.setText(
                f"呼吸率 {rr_str} 次/分  |  旧固件分类: {name} ({conf:.0%})")
        else:
            self._pb_readout.setText(
                f"呼吸率 {rr_str} 次/分  |  EtCO2 {etco2_str} mmHg  |  "
                f"诊断: {name} ({conf:.0%})")
        self._pb_readout.setStyleSheet(
            f"color:#FFFFFF; background-color:{color}; "
            f"font-size:18px; font-weight:bold; padding:6px 12px; border-radius:6px;")
        # 控制栏右侧状态
        self._pb_elapsed.setText(f"t={t:.0f}s")

    def _stop_playback_timer(self):
        """停止播放定时器"""
        if self._playback_timer.isActive():
            self._playback_timer.stop()
        self._playback_playing = False

    def _refresh_history(self):
        """刷新历史数据表格"""
        start = self._hist_start.date().toString("yyyy-MM-dd") + "T00:00:00"
        end = self._hist_end.date().toString("yyyy-MM-dd") + "T23:59:59"
        rows = self.db.query_device_data(start_ts=start, end_ts=end, limit=500)
        self._hist_table.setRowCount(len(rows))
        LABEL_NAMES = {0: "正常", 1: "急促", 2: "屏息", 3: "异常"}
        for i, r in enumerate(rows):
            ts = r['timestamp'][:19] if r['timestamp'] else ""
            self._hist_table.setItem(i, 0, QTableWidgetItem(ts))
            rel = self.display_mode == DISPLAY_MODE_RELATIVE
            self._hist_table.setItem(i, 1, QTableWidgetItem(
                "-" if rel else (f"{r['co2']:.1f}" if r['co2'] else "-")))
            self._hist_table.setItem(i, 2, QTableWidgetItem(
                f"{r['rr']:.1f}" if r['rr'] else "-"))
            self._hist_table.setItem(i, 3, QTableWidgetItem(
                "-" if rel else (f"{r['etco2']:.1f}" if r['etco2'] else "-")))
            label = LABEL_NAMES.get(r['ai_label'], "-") if r['ai_label'] is not None else "-"
            self._hist_table.setItem(i, 4, QTableWidgetItem(label))
            self._hist_table.setItem(i, 5, QTableWidgetItem(
                "legacy" if r['ai_label'] is not None else "-"))
        self._hist_stats.setText(f"共 {len(rows)} 条记录")

    def _refresh_alarm_history(self):
        """刷新告警记录表格"""
        alarms = self.db.query_alarms()
        self._alarm_table.setRowCount(len(alarms))
        for i, a in enumerate(alarms):
            ts = a['timestamp'][:19] if a['timestamp'] else ""
            self._alarm_table.setItem(i, 0, QTableWidgetItem(ts))
            self._alarm_table.setItem(i, 1, QTableWidgetItem(a['alarm_type']))
            self._alarm_table.setItem(i, 2, QTableWidgetItem(
                f"{a['co2_value']:.1f}" if a['co2_value'] else "-"))
            self._alarm_table.setItem(i, 3, QTableWidgetItem(
                f"{a['rr_value']:.1f}" if a['rr_value'] else "-"))
            self._alarm_table.setItem(i, 4, QTableWidgetItem(
                f"{a['confidence']:.0%}" if a['confidence'] else "-"))

    def _export_csv(self):
        """导出历史数据为 CSV 文件（Excel 友好格式）"""
        import csv
        start = self._hist_start.date().toString("yyyy-MM-dd") + "T00:00:00"
        end = self._hist_end.date().toString("yyyy-MM-dd") + "T23:59:59"
        rows = self.db.query_device_data(start_ts=start, end_ts=end, limit=10000)
        if not rows:
            self.log_msg("无数据可导出", level="warn")
            return
        filename = f"export_{self._hist_start.date().toString('yyyyMMdd')}.csv"
        # 弹出保存对话框让用户选择位置（默认当前目录）
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出CSV数据", filename, "CSV 文件 (*.csv)")
        if not filepath:
            return  # 用户取消
        if not filepath.lower().endswith('.csv'):
            filepath += '.csv'
        LABEL_NAMES = {0: "正常", 1: "急促", 2: "屏息", 3: "异常"}
        try:
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(["时间", "相对CO2", "呼吸率", "EtCO2(未标定)",
                            "旧固件分类", "来源"])
                for r in rows:
                    # 时间戳 T 换成空格，Excel 可识别为时间
                    ts = (r['timestamp'] or '').replace('T', ' ')[:19]
                    label = LABEL_NAMES.get(r['ai_label'], "") if r['ai_label'] is not None else ""
                    conf = f"{r['confidence']:.0%}" if r['confidence'] is not None else ""
                    w.writerow([
                        ts,
                        f"{r['co2']:.1f}" if r['co2'] is not None else "",
                        f"{r['rr']:.1f}" if r['rr'] is not None else "",
                        f"{r['etco2']:.1f}" if r['etco2'] is not None else "",
                        label,
                        "legacy" if label else "",
                    ])
            self.log_msg(f"✅ 已导出 {len(rows)} 条记录到 {os.path.basename(filepath)}")
            self.statusBar().showMessage(
                f"已导出 {len(rows)} 条记录: {filepath}")
        except Exception as e:
            self.log_msg(f"导出失败: {e}", level="error")

    def _export_excel(self):
        """导出历史数据为 Excel 文件（含趋势图表）"""
        start = self._hist_start.date().toString("yyyy-MM-dd") + "T00:00:00"
        end = self._hist_end.date().toString("yyyy-MM-dd") + "T23:59:59"
        rows = self.db.query_device_data(start_ts=start, end_ts=end, limit=20000)
        if not rows:
            self.log_msg("无数据可导出", level="warn")
            return

        filename = f"export_{self._hist_start.date().toString('yyyyMMdd')}.xlsx"
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出Excel数据", filename, "Excel 文件 (*.xlsx)")
        if not filepath:
            return
        if not filepath.lower().endswith('.xlsx'):
            filepath += '.xlsx'
        try:
            ExcelExporter.export(rows, filepath)
            self.log_msg(f"✅ 已导出 {len(rows)} 条记录到 {os.path.basename(filepath)}")
            self.statusBar().showMessage(
                f"已导出 {len(rows)} 条记录: {filepath}")
        except Exception as e:
            self.log_msg(f"Excel导出失败: {e}", level="error")

    def _generate_report(self):
        """生成 HTML 诊断报告：填患者信息 → 生成 → 浏览器打开"""
        from PySide6.QtWidgets import QInputDialog
        start = self._hist_start.date().toString("yyyy-MM-dd") + "T00:00:00"
        end = self._hist_end.date().toString("yyyy-MM-dd") + "T23:59:59"

        # 询问患者信息
        name, ok1 = QInputDialog.getText(self, "患者信息", "姓名:")
        if not ok1:
            return
        age, ok2 = QInputDialog.getText(self, "患者信息", "年龄:")
        if not ok2:
            return
        gender, ok3 = QInputDialog.getText(self, "患者信息", "性别:")
        if not ok3:
            return

        patient = {"name": name or "—", "age": age or "—", "gender": gender or "—"}

        try:
            gen = ReportGenerator()
            filename = (f"report_{self._hist_start.date().toString('yyyyMMdd')}"
                        f"_{self._hist_end.date().toString('yyyyMMdd')}.html")
            filepath = os.path.join(app_base_dir(), filename)
            path = gen.save_report(start, end, filepath, patient_info=patient)
            self.log_msg(f"✅ 报告已生成: {filename}")
            self.statusBar().showMessage(f"报告已生成: {filename}，正在打开浏览器...")
            # 用系统默认浏览器打开
            if sys.platform == "win32":
                os.startfile(path)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.log_msg(f"报告生成失败: {e}", level="error")

    def _log_to_ui(self, msg: str):
        """Logger 回调 — 写入 QPlainTextEdit"""
        self.log.appendPlainText(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def log_msg(self, msg: str, level: str = "info", source: str = ""):
        """统一日志入口：分级写入日志文件 + UI 面板"""
        if level == "debug":
            self.logger.debug(msg, source)
        elif level == "warn":
            self.logger.warn(msg, source)
        elif level == "error":
            self.logger.error(msg, source)
        else:
            self.logger.info(msg, source)
        # 同步写入操作日志表（可选，只写重要的）
        if level in ("warn", "error"):
            self.db.insert_operation_log(
                datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), level.upper(), msg, source)

    def closeEvent(self, event):
        """窗口关闭时：每日摘要 → 保存配置 → 停止串口 → 关闭数据库"""
        # 每日统计摘要
        try:
            if hasattr(self, 'db') and self.db is not None:
                summary = self.db.get_daily_summary()
                if summary.get('total', 0) > 0:
                    msg = (f"[每日摘要] {summary['date']}  共{summary['total']}条记录"
                           f" | 正常占比{summary['normal_pct']}%"
                           f" | 平均呼吸率{summary['avg_rr']}次/分"
                           f" | 告警{summary['alarms']}次")
                    self.logger.info(msg, source="summary")
                    self.db.insert_operation_log(
                        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "INFO", msg, "summary")
        except Exception:
            pass
        try:
            self.config.save()
        except Exception:
            pass
        self._teardown_serial_backend()
        try:
            if hasattr(self, 'db') and self.db is not None:
                self.db.close()
        except Exception:
            pass
        super().closeEvent(event)

    def _teardown_serial_backend(self):
        """安全销毁串口后台线程（open() 卡死时也能强制回收，避免程序退出挂起）"""
        try:
            if hasattr(self, '_serial_worker') and self._serial_worker is not None:
                try:
                    self._serial_worker.do_close.emit()
                except Exception:
                    pass
        except Exception:
            pass
        thread = getattr(self, '_serial_thread', None)
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            if not thread.wait(2000):
                # 后台线程卡在原生 open() 中，quit 无法唤醒 → 强制终止（最后手段）
                try:
                    thread.terminate()
                    thread.wait(500)
                except Exception:
                    pass
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._serial_thread = None
        self._serial_worker = None


if __name__ == "__main__":
    if sys.platform == "win32":
        # 打包为 windowed 模式时 sys.stdout/stderr 为 None，需判空
        try:
            if sys.stdout is not None:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if sys.stderr is not None:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CO2MonitorWindow()
    win.show()
    win.raise_()
    sys.exit(app.exec())