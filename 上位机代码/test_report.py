"""
HTML 报告生成模块测试
    python test_report.py
"""
import unittest
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestReportGenerator(unittest.TestCase):
    """报告生成测试"""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="test_report_")
        cls._db_path = os.path.join(cls._tmpdir, "test.db")
        # 用临时 DB 填充数据
        from db_manager import DatabaseManager
        db = DatabaseManager(cls._db_path)
        now = datetime.now()
        for i in range(30):
            ts = (now - timedelta(minutes=i * 2)).strftime("%Y-%m-%dT%H:%M:%S")
            db.insert_device_data(
                ts=ts, co2=38.0, rr=14.0 + (i % 3), etco2=38.5,
                voltage_v=2.5, ai_label=(0 if i % 10 < 8 else 3),
                confidence=0.8)
        db.insert_alarm(
            now.strftime("%Y-%m-%dT%H:%M:%S"), "屏息",
            ai_label=3, confidence=0.9, co2_value=5.0, rr_value=1.0)
        db.close()

        from report_generator import ReportGenerator
        cls.gen = ReportGenerator(cls._db_path)
        cls.start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        cls.end = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_01_generate_valid_html(self):
        """生成合法 HTML"""
        html = self.gen.generate_html(self.start, self.end)
        self.assertIn("<html", html)
        self.assertIn("CO₂ 呼吸健康监测报告", html)
        self.assertIn("总体摘要", html)

    def test_02_contains_patient_info(self):
        """包含患者信息"""
        patient = {"name": "张三", "age": "25", "gender": "男"}
        html = self.gen.generate_html(self.start, self.end, patient_info=patient)
        self.assertIn("张三", html)
        self.assertIn("25", html)
        self.assertIn("男", html)

    def test_03_contains_statistics(self):
        """包含统计数字"""
        html = self.gen.generate_html(self.start, self.end)
        self.assertIn("总记录数", html)
        self.assertIn("平均呼吸率", html)
        self.assertIn("异常告警次数", html)

    def test_04_contains_svg_charts(self):
        """包含 SVG 图表"""
        html = self.gen.generate_html(self.start, self.end)
        self.assertIn("<svg", html)
        self.assertIn("polyline", html)      # 趋势折线
        self.assertIn("rect", html)          # 柱状图/阴影带

    def test_05_contains_alarm_events(self):
        """包含异常事件列表"""
        html = self.gen.generate_html(self.start, self.end)
        self.assertIn("异常事件列表", html)
        self.assertIn("屏息", html)

    def test_06_save_report_file(self):
        """保存报告到文件"""
        out = os.path.join(self._tmpdir, "report.html")
        path = self.gen.save_report(self.start, self.end, out)
        self.assertTrue(os.path.exists(path))
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertGreater(len(content), 500)

    def test_07_no_data_report(self):
        """空数据时段不崩溃"""
        html = self.gen.generate_html("2020-01-01T00:00:00", "2020-01-02T00:00:00")
        self.assertIn("无数据", html)
        self.assertIn("<html", html)

    def test_08_deterministic(self):
        """相同输入生成相同报告"""
        html1 = self.gen.generate_html(self.start, self.end, {"name": "李四"})
        html2 = self.gen.generate_html(self.start, self.end, {"name": "李四"})
        self.assertEqual(html1, html2)


class TestSvgHelpers(unittest.TestCase):
    """SVG 辅助函数测试"""

    def test_downsample_small(self):
        """小数据不降采样"""
        from report_generator import _downsample
        pts = _downsample([1, 2, 3], max_points=10)
        self.assertEqual(len(pts), 3)

    def test_downsample_large(self):
        """大数据降到上限附近"""
        from report_generator import _downsample
        pts = _downsample(list(range(2000)), max_points=100)
        # 100 步采样 + 可能追加的端点，最多 101 点
        self.assertLessEqual(len(pts), 101)
        self.assertGreaterEqual(len(pts), 100)
        # 包含最后一个点
        self.assertEqual(pts[-1][0], 1999)

    def test_fill_forward(self):
        """前值填充"""
        from report_generator import ReportGenerator
        filled = ReportGenerator._fill_forward([1.0, None, None, 5.0])
        self.assertEqual(filled, [1.0, 1.0, 1.0, 5.0])


if __name__ == '__main__':
    print("=" * 60)
    print("HTML 报告生成模块测试")
    print("=" * 60)
    unittest.main(verbosity=2)
