"""
业务逻辑层测试 — 告警 / 趋势 / 历史 / 摘要
    python test_business.py
"""
import unittest
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBusinessLogic(unittest.TestCase):
    """业务逻辑单元测试"""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="test_biz_")
        cls._db_path = os.path.join(cls._tmpdir, "test.db")
        from db_manager import DatabaseManager
        cls.db = DatabaseManager(cls._db_path)
        # 填充一些测试数据
        now = datetime.now()
        for i in range(20):
            ts = (now - timedelta(seconds=i * 5)).strftime("%Y-%m-%dT%H:%M:%S")
            rr = 14.0 + i * 0.5  # 逐步上升
            etco2 = 38.0 - i * 0.3 if i < 10 else 35.0  # 先降后稳
            cls.db.insert_device_data(
                ts=ts, co2=etco2, rr=rr, etco2=etco2,
                voltage_v=2.5, ai_label=(0 if i < 15 else 1),
                confidence=0.8)
        # 填几条告警
        cls.db.insert_alarm(
            (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S"),
            "屏息", ai_label=3, confidence=0.9, co2_value=5.0, rr_value=1.0)
        cls.db.insert_alarm(
            now.strftime("%Y-%m-%dT%H:%M:%S"),
            "呼吸异常", ai_label=4, confidence=0.75, co2_value=48.0, rr_value=11.0)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ── DB 查询方法 ──

    def test_01_query_recent(self):
        """查询最近 N 条数据"""
        rows = self.db.query_recent_device_data(limit=5)
        self.assertEqual(len(rows), 5)
        self.assertIn('co2', rows[0])
        self.assertIn('rr', rows[0])
        self.assertIn('etco2', rows[0])
        self.assertIn('ai_label', rows[0])

    def test_02_query_recent_order(self):
        """最近数据应该按时间倒序"""
        rows = self.db.query_recent_device_data(limit=3)
        # RR 应是递减趋势（因为插入时 i 从低到高，最近的 i=19 RR最高）
        self.assertGreater(rows[0]['rr'], rows[-1]['rr'])

    def test_03_daily_summary(self):
        """每日统计摘要"""
        today = datetime.now().strftime("%Y-%m-%d")
        summary = self.db.get_daily_summary(today)
        self.assertEqual(summary['date'], today)
        self.assertGreater(summary['total'], 0)
        self.assertGreater(summary['alarms'], 0)
        self.assertIn('normal_pct', summary)
        self.assertIn('avg_rr', summary)

    def test_04_daily_summary_empty_day(self):
        """空日期的摘要"""
        summary = self.db.get_daily_summary("2020-01-01")
        self.assertEqual(summary['total'], 0)

    # ── 趋势分析算法（纯函数测试） ──

    def test_05_trend_slope_rising(self):
        """上升趋势：简单线性回归斜率为正"""
        rr_values = [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        n = len(rr_values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(rr_values) / n
        num = sum((i - x_mean) * (rr_values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den
        self.assertGreater(slope, 0.9)  # 接近 +1 的斜率

    def test_06_trend_slope_falling(self):
        """下降趋势"""
        rr_values = [23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0]
        n = len(rr_values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(rr_values) / n
        num = sum((i - x_mean) * (rr_values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den
        self.assertLess(slope, -0.9)

    def test_07_trend_slope_stable(self):
        """稳定趋势：斜率接近 0"""
        rr_values = [14.0, 14.2, 13.8, 14.1, 14.0, 14.3, 13.9, 14.1, 14.0, 14.2]
        n = len(rr_values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(rr_values) / n
        num = sum((i - x_mean) * (rr_values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den
        self.assertLess(abs(slope), 0.5)

    def test_08_etco2_baseline_shift_detection(self):
        """EtCO2 基线偏离检测"""
        etco2_values = [28.0, 27.0, 26.5, 27.5, 28.0]  # 全部低于 30
        abnormal = [e for e in etco2_values if e < 30 or e > 50]
        self.assertEqual(len(abnormal), 5)

    def test_09_etco2_normal(self):
        """EtCO2 正常范围"""
        etco2_values = [37.0, 39.0, 38.5, 40.0, 36.5]
        abnormal = [e for e in etco2_values if e < 30 or e > 50]
        self.assertEqual(len(abnormal), 0)

    # ── 告警确认 ──

    def test_10_alarm_acknowledge(self):
        """告警确认"""
        alarms_before = self.db.query_alarms(acknowledged=0)
        if alarms_before:
            self.db.acknowledge_alarm(alarms_before[0]['id'])
        alarms_after = self.db.query_alarms(acknowledged=1)
        self.assertGreaterEqual(len(alarms_after), 1)


if __name__ == '__main__':
    print("=" * 60)
    print("业务逻辑层测试")
    print("=" * 60)
    unittest.main(verbosity=2)
