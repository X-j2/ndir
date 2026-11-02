"""
数据库 / 配置 / 日志模块测试
    python test_db_log.py
"""
import unittest
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestDatabaseManager(unittest.TestCase):
    """SQLite 数据库模块测试"""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="test_db_")
        cls._db_path = os.path.join(cls._tmpdir, "test.db")
        from db_manager import DatabaseManager
        cls.db = DatabaseManager(cls._db_path)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_01_insert_device_data(self):
        """插入设备采样记录"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.db.insert_device_data(ts, co2=38.5, rr=14.0, etco2=38.0,
                                   voltage_v=2.56, ai_label=0, confidence=0.85)
        rows = self.db.query_device_data(limit=1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['co2'], 38.5, places=1)
        self.assertEqual(rows[0]['ai_label'], 0)

    def test_02_insert_device_data_without_ai(self):
        """插入无 AI 标签的采样"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.db.insert_device_data(ts, co2=40.0)
        rows = self.db.query_device_data(limit=10)
        self.assertTrue(any(r['ai_label'] is None for r in rows))

    def test_03_insert_alarm(self):
        """插入告警记录"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.db.insert_alarm(ts, "屏息", ai_label=3, confidence=0.92,
                             co2_value=5.0, rr_value=2.0)
        alarms = self.db.query_alarms()
        self.assertGreaterEqual(len(alarms), 1)
        self.assertEqual(alarms[-1]['alarm_type'], "屏息")

    def test_04_insert_operation_log(self):
        """插入操作日志"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.db.insert_operation_log(ts, "ERROR", "串口打开失败", "serial")
        self.db.insert_operation_log(ts, "INFO", "模拟启动", "simulator")
        # 操作日志只写入不查询（查询在 log_manager 中实现）
        self.assertTrue(True)  # 不抛异常即通过

    def test_05_query_by_timerange(self):
        """按时间范围查询"""
        now = datetime.now()
        start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        end = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = self.db.query_device_data(start_ts=start, end_ts=end, limit=100)
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)

    def test_06_count_alarms_today(self):
        """今日告警计数"""
        count = self.db.count_alarms_today()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_07_get_statistics(self):
        """统计摘要"""
        stats = self.db.get_statistics()
        self.assertIn('total_samples', stats)
        self.assertIn('label_counts', stats)
        self.assertIn('avg_rr', stats)
        self.assertGreaterEqual(stats['total_samples'], 1)

    def test_08_acknowledge_alarm(self):
        """确认告警"""
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.db.insert_alarm(ts, "急促呼吸", ai_label=1, confidence=0.7)
        alarms = self.db.query_alarms(acknowledged=0)
        if alarms:
            self.db.acknowledge_alarm(alarms[0]['id'])
            acked = self.db.query_alarms(acknowledged=1)
            self.assertGreaterEqual(len(acked), 1)

    def test_09_cleanup(self):
        """清理过期数据"""
        self.db.cleanup_old_data(retention_days=365)  # 不清最近数据
        rows = self.db.query_device_data(limit=1)
        self.assertGreaterEqual(len(rows), 0)


class TestConfigManager(unittest.TestCase):
    """JSON 配置模块测试"""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        cls._cfg_path = os.path.join(cls._tmpdir, "config.json")
        from config_manager import ConfigManager
        cls.cfg = ConfigManager(cls._cfg_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_10_load_defaults(self):
        """加载默认配置"""
        self.assertEqual(self.cfg.get("serial.baud_rate"), 115200)
        self.assertEqual(self.cfg.get("alarm.enabled"), True)
        self.assertEqual(self.cfg.get("ai.inference_interval_sec"), 30)

    def test_11_set_and_get(self):
        """设置并读取配置"""
        self.cfg.set("serial.port", "COM5")
        self.assertEqual(self.cfg.get("serial.port"), "COM5")

        self.cfg.set("alarm.min_confidence", 0.8)
        self.assertEqual(self.cfg.get("alarm.min_confidence"), 0.8)

    def test_12_default_fallback(self):
        """不存在的键返回默认值"""
        self.assertIsNone(self.cfg.get("nonexistent.key"))
        self.assertEqual(self.cfg.get("nonexistent.key", 42), 42)

    def test_13_save_and_reload(self):
        """保存后重新加载"""
        self.cfg.set("serial.baud_rate", 9600)
        self.cfg.save()

        from config_manager import ConfigManager
        cfg2 = ConfigManager(self._cfg_path)
        self.assertEqual(cfg2.get("serial.baud_rate"), 9600)
        self.assertEqual(cfg2.get("serial.port"), "COM5")  # 上一步设置的值

    def test_14_corrupt_file_recovery(self):
        """损坏文件自动恢复为默认值"""
        corrupt_path = os.path.join(self._tmpdir, "corrupt.json")
        with open(corrupt_path, 'w') as f:
            f.write("not valid json {{{")
        from config_manager import ConfigManager
        cfg = ConfigManager(corrupt_path)
        self.assertEqual(cfg.get("serial.baud_rate"), 115200)  # 恢复默认

    def test_15_reset_to_defaults(self):
        """重置为默认值"""
        self.cfg.set("serial.baud_rate", 38400)
        self.cfg.reset_to_defaults()
        self.assertEqual(self.cfg.get("serial.baud_rate"), 115200)

    def test_16_nested_new_key_missing(self):
        """缺失键合并默认值"""
        partial_path = os.path.join(self._tmpdir, "partial.json")
        with open(partial_path, 'w', encoding='utf-8') as f:
            json.dump({"serial": {"port": "COM1"}}, f)
        from config_manager import ConfigManager
        cfg = ConfigManager(partial_path)
        self.assertEqual(cfg.get("serial.port"), "COM1")
        self.assertEqual(cfg.get("serial.baud_rate"), 115200)  # 补全
        self.assertEqual(cfg.get("alarm.enabled"), True)       # 补全


class TestLogManager(unittest.TestCase):
    """分级日志模块测试"""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="test_log_")
        cls._log_dir = cls._tmpdir
        from log_manager import LogManager
        cls.log_mgr = LogManager(log_dir=cls._log_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_17_all_levels_no_crash(self):
        """所有级别日志输出不崩溃"""
        self.log_mgr.debug("调试消息", source="test")
        self.log_mgr.info("运行消息", source="test")
        self.log_mgr.warn("警告消息", source="test")
        self.log_mgr.error("错误消息", source="test")
        self.assertTrue(True)

    def test_18_log_file_created(self):
        """日志文件创建"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self._log_dir, f"co2_monitor_{today}.log")
        self.assertTrue(os.path.exists(log_file),
                        f"日志文件不存在: {log_file}")

    def test_19_log_file_content(self):
        """日志文件有内容"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self._log_dir, f"co2_monitor_{today}.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("运行消息", content)
        # DEBUG 不应出现在文件中（文件级别是 INFO）
        self.assertNotIn("调试消息", content)

    def test_20_get_recent_logs(self):
        """获取最近日志"""
        logs = self.log_mgr.get_recent_logs(3)
        self.assertIsInstance(logs, list)
        self.assertGreater(len(logs), 0)

    def test_21_ui_callback(self):
        """UI 回调注册和调用"""
        captured = []

        def fake_cb(msg):
            captured.append(msg)

        self.log_mgr.set_ui_callback(fake_cb)
        self.log_mgr.info("测试回调", source="ui")
        self.assertGreater(len(captured), 0)
        self.assertIn("测试回调", captured[-1])
        self.assertIn("ui", captured[-1])

    def test_22_cleanup_old_logs(self):
        """清理旧日志不崩溃"""
        self.log_mgr.cleanup_old_logs(max_days=365)
        self.assertTrue(True)


if __name__ == '__main__':
    print("=" * 60)
    print("数据库 / 配置 / 日志模块测试")
    print("=" * 60)
    unittest.main(verbosity=2)
