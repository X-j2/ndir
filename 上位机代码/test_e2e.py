"""
端到端集成测试 — 模拟完整用户流程
    python test_e2e.py

流程：启动窗口 → 模拟运行 → DB写入 → 历史查询 → 告警 → 报告 → CSV → Excel
"""
import unittest
import sys
import os
import tempfile
import shutil
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 用临时 DB 目录隔离测试，避免污染真实数据
TEST_TMP = tempfile.mkdtemp(prefix="test_e2e_")
os.chdir(TEST_TMP)


class TestEndToEnd(unittest.TestCase):
    """全链路集成测试"""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from co2_monitor import CO2MonitorWindow
        # 临时目录下启动窗口（co2_monitor.db 会在临时目录创建）
        cls.win = CO2MonitorWindow()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.win.close()
        except Exception:
            pass
        shutil.rmtree(TEST_TMP, ignore_errors=True)

    def test_01_modules_initialized(self):
        """三模块初始化"""
        w = self.win
        self.assertIsNotNone(w.db)
        self.assertIsNotNone(w.config)
        self.assertIsNotNone(w.logger)
        # 临时 DB 文件应存在
        self.assertTrue(os.path.exists(w.db._db_path), "DB文件不存在")

    def test_02_simulate_runs(self):
        """模拟模式跑 60 帧 → 波形缓冲有数据"""
        w = self.win
        # 启动模拟
        w.sim_running = True
        w.sim_tick = 0
        w.sim_pattern_index = 0
        w.sim_phase = 0.0
        for _ in range(60):
            w.generate_sim_data()
        self.assertGreater(len(w.wave_buffer), 0, "波形缓冲为空")
        self.assertGreater(w.packet_count, 0, "包计数为0")

    def test_03_db_has_data(self):
        """DB 有新写入，rr 非 NULL"""
        rows = self.win.db.query_recent_device_data(limit=10)
        self.assertGreater(len(rows), 0, "DB无数据")
        has_rr = any(r.get('rr') is not None for r in rows)
        self.assertTrue(has_rr, "所有记录 rr 为 NULL")

    def test_04_query_history(self):
        """历史表格数据可查询"""
        rows = self.win.db.query_device_data(limit=20)
        self.assertGreater(len(rows), 0, "历史查询无数据")
        self.assertIn('timestamp', rows[0])
        self.assertIn('co2', rows[0])

    def test_05_legacy_alarm_disabled_in_relative_mode(self):
        """相对模式旧固件分类不写旧告警记录"""
        w = self.win
        from co2_monitor import encode_frame, decode_frame
        before = len(w.db.query_alarms())
        for i in range(3):
            frame_bytes = encode_frame([1500.0] * 5, ai_class=2, rr_per_min=1.0)
            frame = decode_frame(frame_bytes)
            w.latest_probs = {0: 0.02, 1: 0.03, 2: 0.9, 3: 0.05}
            w._apply_frame(frame)
        self.assertEqual(len(w.db.query_alarms()), before,
                         "相对模式旧固件分类不得写旧告警")

    def test_06_html_report(self):
        """HTML 报告生成"""
        from report_generator import ReportGenerator
        gen = ReportGenerator(self.win.db._db_path)
        out = os.path.join(TEST_TMP, "report.html")
        path = gen.save_report("2026-01-01T00:00:00", "2030-12-31T23:59:59", out)
        self.assertTrue(os.path.exists(path), "报告文件不存在")
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn("CO₂ 呼吸健康监测报告", content)
        self.assertIn("总体摘要", content)

    def test_07_csv_export(self):
        """CSV 导出"""
        import csv
        rows = self.win.db.query_device_data(limit=50)
        out = os.path.join(TEST_TMP, "test.csv")
        with open(out, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(["时间", "CO2(mmHg)", "呼吸率", "EtCO2(mmHg)", "AI标签", "置信度"])
            for r in rows:
                ts = (r['timestamp'] or '').replace('T', ' ')[:19]
                w.writerow([ts, r['co2'], r['rr'], r['etco2'], "", ""])
        self.assertTrue(os.path.exists(out))
        with open(out, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1, "CSV无数据行")
        self.assertNotIn('T', lines[1][:20], "时间戳含T")

    def test_08_excel_export(self):
        """Excel 导出（openpyxl）"""
        from excel_exporter import ExcelExporter
        rows = self.win.db.query_device_data(limit=50)
        out = os.path.join(TEST_TMP, "test.xlsx")
        ExcelExporter.export(rows, out)
        self.assertTrue(os.path.exists(out), "Excel文件不存在")
        self.assertGreater(os.path.getsize(out), 1000, "Excel文件过小")
        # 验证可重新打开
        from openpyxl import load_workbook
        wb = load_workbook(out)
        self.assertIn("数据明细", wb.sheetnames)
        self.assertIn("趋势图表", wb.sheetnames)
        ws = wb["数据明细"]
        self.assertEqual(ws.cell(row=1, column=1).value, "时间")

    def test_09_trend_playback_load(self):
        """趋势回放加载数据"""
        w = self.win
        w._pb_start.setDate(__import__('datetime').datetime.now().date())
        w._pb_end.setDate(__import__('datetime').datetime.now().date())
        w._load_playback_data()
        self.assertGreaterEqual(len(w._playback_rows), 0)  # 不崩溃即可


if __name__ == '__main__':
    print("=" * 60)
    print("端到端集成测试（全链路）")
    print(f"临时目录: {TEST_TMP}")
    print("=" * 60)
    unittest.main(verbosity=2)
