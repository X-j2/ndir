"""
模拟模式 AI 分类回归测试
    python test_sim_ai.py

背景：曾经出现"模拟模式下一旦预测呼吸异常就一直不变"的 bug。根因有两条：
  1. SIM_PATTERNS 单轮时长(58s) < AI 推理窗口(90s) → 每个窗口混入全部 4 种模式，
     合成出不属于任何单类的波形，模型结论恒定。
  2. 模拟波形形态不像真实呼吸波(缺呼气平台、基线偏高) → 特征偏离训练分布。

本测试锁定这两点：
  - 每个模式时长必须 >= AI 推理窗口（保证窗口落在单一模式内）
  - 每个模式生成的 90s 波形，模型预测必须命中其 truth 标签（多随机种子）
"""
import unittest
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSimPatternDuration(unittest.TestCase):
    """模拟模式时长约束（防回归：窗口跨模式导致结论恒定）"""

    def test_duration_covers_ai_window(self):
        """每个模式的持续时长必须 >= AI 推理窗口（90s）"""
        from co2_monitor import SIM_PATTERNS, AI_WINDOW_SECONDS
        for p in SIM_PATTERNS:
            self.assertGreaterEqual(
                p["duration"], AI_WINDOW_SECONDS,
                f"模式「{p['name']}」时长 {p['duration']}s < AI 窗口 "
                f"{AI_WINDOW_SECONDS}s，90s 窗口会横跨多个模式导致诊断结论失真")
            # 切换时会清空推理窗口，前 90s 处于预热，需留出有效诊断时间
            self.assertGreaterEqual(
                p["duration"], AI_WINDOW_SECONDS * 2,
                f"模式「{p['name']}」时长 {p['duration']}s 不足窗口的 2 倍，"
                "预热后几乎没有有效诊断展示时间")

    def test_patterns_have_capnogram_params(self):
        """模式参数采用真实呼吸波形语义（rr/etco2/base/f3）"""
        from co2_monitor import SIM_PATTERNS
        for p in SIM_PATTERNS:
            for key in ("rr", "etco2", "base", "f3", "truth", "name"):
                self.assertIn(key, p, f"模式「{p['name']}」缺少参数 {key}")


class TestCapnogramShape(unittest.TestCase):
    """波形形态测试（无需模型）"""

    def test_phase_range(self):
        """一个呼吸周期内波形非负且峰值接近 etco2"""
        from co2_monitor import CO2MonitorWindow, SIM_PATTERNS
        for pat in SIM_PATTERNS:
            vals = [CO2MonitorWindow._capnogram(ph / 100.0, pat) for ph in range(100)]
            self.assertGreaterEqual(min(vals), -1e-6,
                                    f"模式「{pat['name']}」出现负值")
            self.assertLessEqual(max(vals), pat["etco2"] + 1e-6,
                                 f"模式「{pat['name']}」峰值超过 etco2")

    def test_has_expiratory_plateau(self):
        """有呼吸的模式应存在呼气平台（真实波形特征）"""
        from co2_monitor import CO2MonitorWindow, SIM_PATTERNS
        for pat in SIM_PATTERNS:
            if pat["rr"] < 3:
                continue  # 屏息无平台
            vals = [CO2MonitorWindow._capnogram(ph / 200.0, pat) for ph in range(200)]
            near_top = sum(1 for v in vals if v >= pat["etco2"] * 0.90) / len(vals)
            self.assertGreater(near_top, 0.10,
                               f"模式「{pat['name']}」缺乏呼气平台（占比仅 {near_top:.2f}）")


class TestSimModeClassification(unittest.TestCase):
    """端到端：模拟波形 → 特征 → ONNX → 必须命中 truth 标签"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from co2_monitor import CO2MonitorWindow
        cls.win = CO2MonitorWindow()
        if not cls.win.model_loaded:
            raise unittest.SkipTest("ONNX 模型未加载，跳过模拟分类测试")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass

    def _simulate(self, pattern, seconds=140, seed=0):
        """复刻 generate_sim_data 的采样过程，返回 90s 窗口波形"""
        random.seed(seed)
        phase = 0.0
        buf = []
        for _ in range(seconds * 5):
            for _j in range(5):
                phase += (pattern["rr"] / 60.0) * 0.040
                if phase >= 1.0:
                    phase -= 1.0
                c = self.win._capnogram(phase, pattern) + random.gauss(0, 0.8)
                buf.append(c)
        return buf[-2250:]

    def test_each_mode_classified_as_truth(self):
        """4 种模式各 6 个随机种子，模型预测必须全部命中 truth"""
        from co2_monitor import SIM_PATTERNS, LABEL_MAP
        for pat in SIM_PATTERNS:
            hits = 0
            for seed in range(6):
                self.win.ai_data_buffer.clear()
                self.win.ai_data_buffer.extend(self._simulate(pat, seed=seed))
                label, _rr, _probs = self.win.run_onnx_inference()
                if label == pat["truth"]:
                    hits += 1
            self.assertEqual(
                hits, 6,
                f"模式「{pat['name']}」(truth={pat['truth']}) "
                f"仅 {hits}/6 命中，应为 6/6")


class TestModeBoundaryReset(unittest.TestCase):
    """模式切换必须清空推理窗口（防跨模式混合波形导致误判）"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from co2_monitor import CO2MonitorWindow
        cls.win = CO2MonitorWindow()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass

    def test_reset_clears_buffer_and_result(self):
        """_reset_ai_window 应清空缓冲与上次推理结果"""
        w = self.win
        w.ai_data_buffer.extend([30.0] * 2250)
        w.latest_probs = {0: 0.9}
        w.latest_label = 0
        w.latest_local_label = 0
        w._alarm_consecutive = 3
        w._reset_ai_window()
        self.assertEqual(len(w.ai_data_buffer), 0, "AI 缓冲未清空")
        self.assertIsNone(w.latest_probs, "上次概率未清空")
        self.assertIsNone(w.latest_label, "上次标签未清空")
        self.assertEqual(w._alarm_consecutive, 0, "连续异常计数未重置")

    def test_warmup_after_switch(self):
        """切换清空后应进入预热态（不给出跨模式结论）"""
        w = self.win
        w.ai_data_buffer.extend([30.0] * 2250)
        self.assertFalse(w._is_warming_up())
        w._reset_ai_window()
        self.assertTrue(w._is_warming_up(), "切换后应处于预热中")


class TestWarmupAndAlarmThrottle(unittest.TestCase):
    """预热判定 + 告警限频（防刷屏）"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from co2_monitor import CO2MonitorWindow
        cls.win = CO2MonitorWindow()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass

    def _fill_window(self, points):
        self.win.ai_data_buffer.clear()
        for _ in range(points):
            self.win.ai_data_buffer.append(30.0)

    def test_warmup_detection(self):
        """窗口未满 → 预热中；满 → 非预热"""
        from co2_monitor import AI_WINDOW_POINTS
        self._fill_window(100)
        self.assertTrue(self.win._is_warming_up(), "少量数据应判为预热中")
        self._fill_window(AI_WINDOW_POINTS)
        self.assertFalse(self.win._is_warming_up(), "满窗不应判为预热中")

    def test_no_alarm_during_warmup(self):
        """预热期不写入告警记录"""
        from co2_monitor import encode_frame, decode_frame
        self._fill_window(300)          # 12s，远未满窗
        self.win._alarm_consecutive = 0
        self.win._last_ai_label = None
        self.win._last_alarm_time = None
        before = len(self.win.db.query_alarms())
        for _ in range(4):
            f = decode_frame(encode_frame([1500.0] * 5, ai_class=2, rr_per_min=1.0))
            self.win.latest_probs = {0: 0.02, 1: 0.03, 2: 0.9, 3: 0.05}
            self.win._apply_frame(f)
        self.assertEqual(len(self.win.db.query_alarms()), before,
                         "预热期不应写入告警记录")

    def test_legacy_alarm_chain_disabled_in_relative_mode(self):
        """旧固件分类不再写入旧告警记录；新告警只由事件引擎输出"""
        from co2_monitor import encode_frame, decode_frame, AI_WINDOW_POINTS
        self._fill_window(AI_WINDOW_POINTS)
        before = len(self.win.db.query_alarms())
        for _ in range(40):
            f = decode_frame(encode_frame([1500.0] * 5, ai_class=2, rr_per_min=1.0))
            self.win.latest_probs = {0: 0.02, 1: 0.03, 2: 0.9, 3: 0.05}
            self.win._apply_frame(f)
        self.assertEqual(len(self.win.db.query_alarms()), before,
                         "相对模式旧固件分类不得写旧告警")


if __name__ == "__main__":
    unittest.main(verbosity=2)
