"""
SQLite 数据库模块 — 设备数据、告警记录、操作日志的存储与查询
    from db_manager import DatabaseManager
    db = DatabaseManager("co2_monitor.db")
    db.insert_device_data(ts, co2, rr, etco2, voltage, label, conf)
    db.close()
"""
import sqlite3
import os
from datetime import datetime, timedelta
from threading import Lock

from paths import app_base_dir


CREATE_DEVICE_DATA = """
CREATE TABLE IF NOT EXISTS device_data (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    co2        REAL,
    rr         REAL,
    etco2      REAL,
    voltage_v  REAL,
    ai_label   INTEGER,
    confidence REAL
);
"""

CREATE_ALARM_RECORDS = """
CREATE TABLE IF NOT EXISTS alarm_records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    alarm_type TEXT    NOT NULL,
    ai_label   INTEGER,
    confidence REAL,
    co2_value  REAL,
    rr_value   REAL,
    acknowledged INTEGER DEFAULT 0
);
"""

CREATE_EVENT_RECORDS = """
CREATE TABLE IF NOT EXISTS event_records (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT    NOT NULL,
    session_id         TEXT,
    mode               TEXT,
    scale              TEXT    NOT NULL,
    state              TEXT    NOT NULL,
    label              TEXT,
    estimated_onset    REAL,
    first_prompt_time  REAL,
    first_active_time  REAL,
    end_time           REAL,
    source             TEXT,
    confidence         REAL,
    acknowledged       INTEGER DEFAULT 0,
    episode_id         INTEGER,
    transition_seq     INTEGER
);
"""

CREATE_OPERATION_LOGS = """
CREATE TABLE IF NOT EXISTS operation_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    level      TEXT    NOT NULL,
    message    TEXT    NOT NULL,
    source     TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_device_ts ON device_data(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_alarm_ts ON alarm_records(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_oplog_ts ON operation_logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_event_ts ON event_records(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_oplog_level ON operation_logs(level);",
]

INSERT_DEVICE = """
INSERT INTO device_data (timestamp, co2, rr, etco2, voltage_v, ai_label, confidence)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

INSERT_ALARM = """
INSERT INTO alarm_records (timestamp, alarm_type, ai_label, confidence, co2_value, rr_value)
VALUES (?, ?, ?, ?, ?, ?);
"""

INSERT_EVENT = """
INSERT INTO event_records
(timestamp, session_id, mode, scale, state, label, estimated_onset,
 first_prompt_time, first_active_time, end_time, source, confidence,
 episode_id, transition_seq)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

INSERT_OPLOG = """
INSERT INTO operation_logs (timestamp, level, message, source)
VALUES (?, ?, ?, ?);
"""


class DatabaseManager:
    """CO2 监控系统 SQLite 数据库管理器"""

    def __init__(self, db_path: str = "co2_monitor.db"):
        self._db_path = os.path.join(app_base_dir(), db_path)
        self._lock = Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            self._conn.execute(CREATE_DEVICE_DATA)
            self._conn.execute(CREATE_ALARM_RECORDS)
            self._conn.execute(CREATE_EVENT_RECORDS)
            self._ensure_event_columns()
            self._conn.execute(CREATE_OPERATION_LOGS)
            for idx_sql in CREATE_INDEXES:
                self._conn.execute(idx_sql)
            self._conn.commit()

    def _ensure_event_columns(self):
        """Add Stage 3 snapshot identifiers to pre-existing databases."""
        existing = {row[1] for row in
                    self._conn.execute("PRAGMA table_info(event_records)")}
        for name, decl in (("episode_id", "INTEGER"),
                           ("transition_seq", "INTEGER")):
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE event_records ADD COLUMN {name} {decl}")

    # ── 插入 ──

    def insert_device_data(self, ts: str, co2: float, rr: float = None,
                           etco2: float = None, voltage_v: float = None,
                           ai_label: int = None, confidence: float = None):
        """插入一条设备采样记录"""
        try:
            with self._lock:
                self._conn.execute(INSERT_DEVICE,
                                   (ts, co2, rr, etco2, voltage_v, ai_label, confidence))
                self._conn.commit()
        except Exception:
            pass  # DB 写失败不应影响主流程

    def insert_alarm(self, ts: str, alarm_type: str, ai_label: int = None,
                     confidence: float = None, co2_value: float = None,
                     rr_value: float = None):
        """插入一条告警记录"""
        try:
            with self._lock:
                self._conn.execute(INSERT_ALARM,
                                   (ts, alarm_type, ai_label, confidence, co2_value, rr_value))
                self._conn.commit()
        except Exception:
            pass

    def insert_event_record(self, ts: str, session_id: str, mode: str,
                            scale: str, state: str, label: str,
                            estimated_onset: float = None,
                            first_prompt_time: float = None,
                            first_active_time: float = None,
                            end_time: float = None,
                            source: str = "stage3_event_engine",
                            confidence: float = None,
                            episode_id: int = None,
                            transition_seq: int = None):
        """Insert a Stage 3 engine event (separate from legacy alarm_records)."""
        try:
            with self._lock:
                self._conn.execute(INSERT_EVENT, (
                    ts, session_id, mode, scale, state, label,
                    estimated_onset, first_prompt_time, first_active_time,
                    end_time, source, confidence, episode_id,
                    transition_seq))
                self._conn.commit()
        except Exception:
            pass

    def query_event_records(self, limit: int = 200) -> list:
        sql = ("SELECT * FROM event_records ORDER BY id DESC LIMIT ?")
        try:
            with self._lock:
                cur = self._conn.execute(sql, (limit,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []

    def insert_operation_log(self, ts: str, level: str, message: str, source: str = ""):
        """插入一条操作日志"""
        try:
            with self._lock:
                self._conn.execute(INSERT_OPLOG, (ts, level, message, source))
                self._conn.commit()
        except Exception:
            pass

    # ── 查询 ──

    def query_device_data(self, start_ts: str = None, end_ts: str = None,
                          limit: int = 1000) -> list:
        """查询设备采样数据"""
        sql = "SELECT * FROM device_data"
        params = []
        if start_ts or end_ts:
            conditions = []
            if start_ts:
                conditions.append("timestamp >= ?")
                params.append(start_ts)
            if end_ts:
                conditions.append("timestamp <= ?")
                params.append(end_ts)
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def query_alarms(self, start_ts: str = None, end_ts: str = None,
                     acknowledged: int = None) -> list:
        """查询告警记录"""
        sql = "SELECT * FROM alarm_records"
        conditions = []
        params = []
        if start_ts:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= ?")
            params.append(end_ts)
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(acknowledged)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC"
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def count_alarms_today(self) -> int:
        """今日告警数量"""
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM alarm_records WHERE timestamp >= ?", (today,))
                return cur.fetchone()[0]
        except Exception:
            return 0

    def query_recent_device_data(self, limit: int = 10) -> list:
        """查询最近 N 条设备采样数据（用于趋势分析）"""
        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT co2, rr, etco2, ai_label, confidence, timestamp "
                    "FROM device_data ORDER BY id DESC LIMIT ?", (limit,))
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def get_daily_summary(self, date_str: str = None) -> dict:
        """获取某天的统计摘要"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        # 计算次日日期用于范围查询
        next_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        stats = {'date': date_str, 'total': 0, 'alarms': 0, 'normal_pct': 0}
        try:
            with self._lock:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM device_data WHERE timestamp >= ? AND timestamp < ?",
                    (date_str, next_date)).fetchone()[0]
                stats['total'] = total
                if total > 0:
                    normal = self._conn.execute(
                        "SELECT COUNT(*) FROM device_data WHERE timestamp >= ? AND timestamp < ? AND ai_label=0",
                        (date_str, next_date)).fetchone()[0]
                    stats['normal_pct'] = round(normal / total * 100, 1)
                    avg_rr = self._conn.execute(
                        "SELECT AVG(rr) FROM device_data WHERE timestamp >= ? AND timestamp < ? AND rr IS NOT NULL",
                        (date_str, next_date)).fetchone()[0]
                    stats['avg_rr'] = round(avg_rr, 1) if avg_rr else 0.0
                stats['alarms'] = self._conn.execute(
                    "SELECT COUNT(*) FROM alarm_records WHERE timestamp >= ? AND timestamp < ?",
                    (date_str, next_date)).fetchone()[0]
        except Exception:
            pass
        return stats

    def get_statistics(self, start_ts: str = None, end_ts: str = None) -> dict:
        """统计摘要：各分类数量、平均呼吸率等"""
        stats = {}
        try:
            with self._lock:
                # 总采样数
                sql = "SELECT COUNT(*) FROM device_data"
                params = []
                if start_ts or end_ts:
                    conds = []
                    if start_ts:
                        conds.append("timestamp >= ?")
                        params.append(start_ts)
                    if end_ts:
                        conds.append("timestamp <= ?")
                        params.append(end_ts)
                    sql += " WHERE " + " AND ".join(conds)
                stats['total_samples'] = self._conn.execute(sql, params).fetchone()[0]

                # 各 AI 分类计数
                sql2 = ("SELECT ai_label, COUNT(*) FROM device_data "
                        "WHERE ai_label IS NOT NULL")
                if start_ts:
                    sql2 += " AND timestamp >= ?"
                if end_ts:
                    sql2 += " AND timestamp <= ?"
                cur2 = self._conn.execute(sql2, params)
                stats['label_counts'] = dict(cur2.fetchall())

                # 平均呼吸率
                sql3 = ("SELECT AVG(rr) FROM device_data WHERE rr IS NOT NULL")
                if start_ts:
                    sql3 += " AND timestamp >= ?"
                if end_ts:
                    sql3 += " AND timestamp <= ?"
                avg_rr = self._conn.execute(sql3, params).fetchone()[0]
                stats['avg_rr'] = round(avg_rr, 1) if avg_rr else 0.0
        except Exception:
            pass
        return stats

    def acknowledge_alarm(self, alarm_id: int):
        """确认告警"""
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE alarm_records SET acknowledged=1 WHERE id=?", (alarm_id,))
                self._conn.commit()
        except Exception:
            pass

    # ── 维护 ──

    def cleanup_old_data(self, retention_days: int = 30):
        """清理过期数据"""
        cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with self._lock:
                self._conn.execute("DELETE FROM device_data WHERE timestamp < ?", (cutoff,))
                self._conn.execute("DELETE FROM alarm_records WHERE timestamp < ?", (cutoff,))
                self._conn.execute("DELETE FROM operation_logs WHERE timestamp < ?", (cutoff,))
                self._conn.execute("VACUUM;")
                self._conn.commit()
        except Exception:
            pass

    def close(self):
        """关闭数据库连接"""
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
