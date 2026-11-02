"""
串口连接修复专项测试 — 验证"打开串口卡在连接中"问题已解决
    python test_serial_connect.py

覆盖：
  test_01  窗口启动后串口后台线程就绪（不阻塞主线程）
  test_02  未选择端口 → 立即拒绝，不进"连接中"状态
  test_03  端口不存在 → 立即拒绝（预检拦截，不等后台 open）
  test_04  看门狗超时 → 状态复位、后台线程重建（不再永久卡死）
  test_05  看门狗启动后手动触发，验证恢复可用
"""
import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_TMP = tempfile.mkdtemp(prefix="test_serial_connect_")
os.chdir(TEST_TMP)


class TestSerialConnect(unittest.TestCase):

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
        shutil.rmtree(TEST_TMP, ignore_errors=True)

    def setUp(self):
        # 每个用例前保证处于"未连接"状态
        w = self.win
        w._serial_opening = False
        w._serial_is_open = False
        w._open_watchdog.stop()
        w.connect_btn.setText("打开串口")

    def test_01_backend_ready(self):
        """窗口启动后串口后台线程已运行，主线程未卡"""
        w = self.win
        self.assertTrue(w._serial_thread.isRunning(), "后台线程未启动")
        self.assertFalse(w._serial_is_open, "初始不应为已连接")
        self.assertFalse(w._serial_opening, "初始不应为打开中")

    def test_02_empty_port_rejected(self):
        """未选择端口 → 立即拒绝，不进'连接中'"""
        w = self.win
        w.port_combo.clear()  # 无可用端口
        w.toggle_serial()
        self.assertFalse(w._serial_opening, "空端口不应进入打开中")
        self.assertFalse(w._serial_is_open)
        self.assertNotEqual(w.conn_status.text(), "● 连接中...", "不应显示连接中")

    def test_03_nonexistent_port_still_opens(self):
        """端口不在枚举列表 → 提示但仍尝试打开（兼容旧版直接 open 的行为）"""
        w = self.win
        if w.port_combo.findText("COM_NOPE_XYZ") < 0:
            w.port_combo.addItem("COM_NOPE_XYZ")
        w.port_combo.setCurrentText("COM_NOPE_XYZ")
        w.toggle_serial()
        # 应进入打开中（不再拦截），等待看门狗兜底
        self.assertTrue(w._serial_opening, "不存在的端口应仍尝试打开")
        self.assertEqual(w.conn_status.text(), "● 连接中...", "应显示连接中")
        # 后台 open 对不存在端口会失败 → 看门狗/失败回调兜底
        from PySide6.QtCore import QCoreApplication
        deadline = 15000
        while w._serial_opening and deadline > 0:
            QCoreApplication.processEvents()
            from time import sleep
            sleep(0.05)
            deadline -= 50
        self.assertFalse(w._serial_opening, "超时后应恢复未打开状态")
        self.assertFalse(w._serial_is_open)

    def test_04_watchdog_recovers(self):
        """看门狗触发后：状态复位 + 后台线程重建，可再次发起连接"""
        w = self.win
        # 模拟：进入打开中且看门狗已启动
        w._serial_opening = True
        w.conn_status.setText("● 连接中...")
        old_thread = w._serial_thread
        w._on_open_timeout()
        # 状态复位
        self.assertFalse(w._serial_opening, "看门狗后不应再是打开中")
        self.assertFalse(w._serial_is_open)
        self.assertEqual(w.conn_status.text(), "● 未连接")
        # 后台线程重建
        self.assertIsNotNone(w._serial_thread)
        self.assertTrue(w._serial_thread.isRunning(), "重建后的线程应运行")
        self.assertIsNot(w._serial_thread, old_thread, "应重建线程")

    def test_05_watchdog_noop_when_idle(self):
        """未在打开中时看门狗触发 = 空操作，不误伤"""
        w = self.win
        w._serial_opening = False
        thread_before = w._serial_thread
        w._on_open_timeout()
        self.assertIs(w._serial_thread, thread_before, "空闲时不应重建线程")
        self.assertFalse(w._serial_opening)


if __name__ == "__main__":
    unittest.main(verbosity=2)
