"""
HTML 诊断报告生成模块 — 零第三方依赖
    from report_generator import ReportGenerator
    gen = ReportGenerator("co2_monitor.db")
    path = gen.save_report("2026-08-01T00:00:00", "2026-08-04T23:59:59", "report.html",
                           patient_info={"name": "张三", "age": "25", "gender": "男"})
"""
import os
from datetime import datetime

from db_manager import DatabaseManager


# ── 标签映射（与 co2_monitor.py 一致，v2.2：4 类，键 0/1/2/3）──
LABEL_NAMES = {0: "正常呼吸", 1: "急促呼吸", 2: "屏息", 3: "呼吸异常"}
LABEL_COLORS = {0: "#00CC66", 1: "#FF9933", 2: "#FF4444", 3: "#FFDD44"}

# 正常范围
RR_NORMAL = (12, 20)      # 呼吸率正常范围（次/分）
ETCO2_NORMAL = (35, 45)   # EtCO2 正常范围（mmHg）

MAX_PLOT_POINTS = 500     # 趋势图最多绘制点数（降采样上限）


def _svg_polyline(points, color, width=2, fill="none"):
    """生成 SVG 折线元素"""
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline points="{pts_str}" fill="{fill}" '
            f'stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round"/>')


def _svg_shadow_band(x0, x1, y0, y1, color, alpha=0.12):
    """生成 SVG 半透明阴影带（正常范围）"""
    return (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{max(x1-x0,0):.1f}" '
            f'height="{max(y1-y0,0):.1f}" fill="{color}" fill-opacity="{alpha}"/>')


def _downsample(values, max_points=MAX_PLOT_POINTS):
    """等间隔降采样，保留端点"""
    n = len(values)
    if n <= max_points:
        return list(enumerate(values))
    step = n / max_points
    result = []
    for i in range(max_points):
        idx = min(n - 1, int(i * step))
        result.append((idx, values[idx]))
    # 确保包含最后一个点
    if result[-1][0] != n - 1:
        result.append((n - 1, values[-1]))
    return result


class ReportGenerator:
    """从数据库生成 HTML 诊断报告"""

    def __init__(self, db_path: str = "co2_monitor.db"):
        self._db = DatabaseManager(db_path)

    # ── 公开接口 ──

    def generate_html(self, start_ts: str, end_ts: str,
                      patient_info: dict = None) -> str:
        """生成完整 HTML 报告字符串"""
        rows = self._db.query_device_data(start_ts=start_ts, end_ts=end_ts, limit=20000)
        alarms = self._db.query_alarms(start_ts=start_ts, end_ts=end_ts)
        stats = self._db.get_statistics(start_ts=start_ts, end_ts=end_ts)

        # 正序排列
        rows.reverse()

        if not rows:
            return self._render_no_data(start_ts, end_ts, patient_info)

        # 计算摘要
        total = stats.get('total_samples', len(rows))
        label_counts = stats.get('label_counts', {})
        avg_rr = stats.get('avg_rr', 0.0)
        normal_count = sum(v for k, v in label_counts.items() if k == 0)
        normal_pct = round(normal_count / total * 100, 1) if total else 0.0
        alarm_count = len(alarms)

        # 提取趋势数据（正序）
        rr_vals = []
        etco2_vals = []
        for r in rows:
            rr_vals.append(r['rr'] if r['rr'] is not None else None)
            etco2_vals.append(r['etco2'] if r['etco2'] is not None else None)

        # 前值填充 None 用于绘图
        rr_filled = self._fill_forward(rr_vals)
        etco2_filled = self._fill_forward(etco2_vals)

        # 生成 SVG 图表
        rr_svg = self._render_trend_svg(rr_filled, "呼吸率 (次/分)",
                                        normal_range=RR_NORMAL, color="#3399FF")
        etco2_svg = self._render_trend_svg(etco2_filled, "EtCO2 (mmHg)",
                                           normal_range=ETCO2_NORMAL, color="#FF9933")
        dist_svg = self._render_distribution_svg(label_counts)

        # 异常事件列表（取最近 50 条）
        alarm_rows = ""
        for a in alarms[:50]:
            ts = (a['timestamp'] or '')[:19]
            co2_str = f"{a['co2_value']:.1f}" if a['co2_value'] is not None else "-"
            rr_str = f"{a['rr_value']:.1f}" if a['rr_value'] is not None else "-"
            alarm_rows += (
                f"<tr><td>{ts}</td><td>{a['alarm_type']}</td>"
                f"<td>{co2_str}</td><td>{rr_str}</td></tr>"
            )
        if not alarm_rows:
            alarm_rows = '<tr><td colspan="4">无异常事件</td></tr>'

        # 数据抽样表格（前 20 条）
        sample_rows = ""
        for r in rows[:20]:
            ts = (r['timestamp'] or '')[:19]
            label = LABEL_NAMES.get(r['ai_label'], "-") if r['ai_label'] is not None else "-"
            co2_str = f"{r['co2']:.1f}" if r['co2'] is not None else "-"
            rr_str = f"{r['rr']:.1f}" if r['rr'] is not None else "-"
            etco2_str = f"{r['etco2']:.1f}" if r['etco2'] is not None else "-"
            sample_rows += (
                f"<tr><td>{ts}</td><td>{co2_str}</td>"
                f"<td>{rr_str}</td><td>{etco2_str}</td>"
                f"<td>{label}</td></tr>"
            )

        patient = patient_info or {}
        name = patient.get('name', '—')
        age = patient.get('age', '—')
        gender = patient.get('gender', '—')

        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_disp = start_ts[:10]
        end_disp = end_ts[:10]

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>CO2 呼吸健康监测报告</title>
<style>
    body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; color: #222; }}
    h1 {{ text-align: center; color: #1a1a1a; border-bottom: 3px solid #00CC66; padding-bottom: 12px; }}
    h2 {{ color: #333; margin-top: 28px; }}
    .patient {{ background: #f5f7f6; border-radius: 8px; padding: 12px 16px; margin: 16px 0; }}
    .patient span {{ margin-right: 28px; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .card {{ flex: 1; min-width: 140px; background: #f0f4f1; border-radius: 8px;
             padding: 12px 16px; text-align: center; border-top: 4px solid #00CC66; }}
    .card .num {{ font-size: 26px; font-weight: bold; color: #1a1a1a; }}
    .card .lbl {{ font-size: 12px; color: #777; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    th {{ background: #eef5f0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 32px;
              border-top: 1px solid #ddd; padding-top: 12px; }}
    svg {{ max-width: 100%; height: auto; }}
    .no-data {{ text-align: center; padding: 60px; color: #999; }}
</style>
</head>
<body>
<h1>🫁 CO₂ 呼吸健康监测报告</h1>

<div class="patient">
    <span><b>患者:</b> {name}</span>
    <span><b>年龄:</b> {age}</span>
    <span><b>性别:</b> {gender}</span>
</div>

<p><b>监测时段:</b> {start_disp} ~ {end_disp}</p>

<h2>总体摘要</h2>
<div class="cards">
    <div class="card"><div class="num">{total}</div><div class="lbl">总记录数</div></div>
    <div class="card"><div class="num">{normal_pct}%</div><div class="lbl">正常占比</div></div>
    <div class="card"><div class="num">{avg_rr}</div><div class="lbl">平均呼吸率(次/分)</div></div>
    <div class="card"><div class="num">{alarm_count}</div><div class="lbl">异常告警次数</div></div>
</div>

<h2>① 呼吸率趋势</h2>
<p style="font-size:12px;color:#999;">阴影带为正常范围 {RR_NORMAL[0]}-{RR_NORMAL[1]} 次/分</p>
{rr_svg}

<h2>② EtCO₂ 趋势</h2>
<p style="font-size:12px;color:#999;">阴影带为正常范围 {ETCO2_NORMAL[0]}-{ETCO2_NORMAL[1]} mmHg</p>
{etco2_svg}

<h2>③ AI 诊断分布</h2>
{dist_svg}

<h2>④ 异常事件列表</h2>
<table>
    <tr><th>时间</th><th>类型</th><th>CO₂(mmHg)</th><th>呼吸率</th></tr>
    {alarm_rows}
</table>

<h2>⑤ 数据抽样（前 {min(len(rows), 20)} 条）</h2>
<table>
    <tr><th>时间</th><th>CO₂(mmHg)</th><th>呼吸率</th><th>EtCO₂</th><th>AI诊断</th></tr>
    {sample_rows}
</table>

<div class="footer">报告生成时间: {gen_time} | 由 CO2 呼吸波形监控系统自动生成</div>
</body>
</html>"""

    def save_report(self, start_ts: str, end_ts: str, output_path: str,
                    patient_info: dict = None) -> str:
        """生成 HTML 并写入文件，返回文件路径"""
        html = self.generate_html(start_ts, end_ts, patient_info)
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path

    # ── 内部辅助 ──

    @staticmethod
    def _fill_forward(values):
        """前值填充 None"""
        result = []
        last = 0.0
        for v in values:
            if v is not None:
                last = float(v)
            result.append(last)
        return result

    def _render_trend_svg(self, values, title, normal_range, color,
                          width=800, height=200):
        """渲染趋势折线 SVG，含正常范围阴影带"""
        if not values:
            return f'<svg width="{width}" height="{height}"><text x="10" y="20">{title}: 无数据</text></svg>'

        n = len(values)
        max_v = max(max(values), normal_range[1] * 1.2) if values else 50
        min_v = min(min(values), normal_range[0] * 0.5) if values else 0
        if max_v - min_v < 1:
            max_v = min_v + 1

        pad = 30
        plot_w = width - 2 * pad
        plot_h = height - 2 * pad

        def x_of(i):
            return pad + (i / max(n - 1, 1)) * plot_w

        def y_of(v):
            return pad + (1 - (v - min_v) / (max_v - min_v)) * plot_h

        # 阴影带
        band_y0 = y_of(normal_range[1])
        band_y1 = y_of(normal_range[0])
        bands = _svg_shadow_band(pad, width - pad, band_y0, band_y1 - band_y0, color)

        # 网格线
        grid = ""
        for gv in range(int(min_v // 10 * 10), int(max_v) + 10, 10):
            gy = y_of(gv)
            grid += (f'<line x1="{pad}" y1="{gy:.1f}" x2="{width-pad}" y2="{gy:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
            grid += f'<text x="4" y="{gy+4:.1f}" font-size="10" fill="#999">{gv}</text>'

        # 折线（降采样）
        pts = _downsample(values)
        poly = _svg_polyline([(x_of(i), y_of(v)) for i, v in pts], color)

        # 边框
        frame = (f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" '
                 f'fill="none" stroke="#ccc" stroke-width="1"/>')

        return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                f'{grid}{bands}{poly}{frame}</svg>')

    def _render_distribution_svg(self, label_counts, width=800, height=200):
        """渲染 AI 诊断分布柱状图（v2.2：4 类，键 0/1/2/3）"""
        keys = sorted(LABEL_NAMES.keys())  # [0,1,2,3]
        labels = [LABEL_NAMES.get(k, f"类{k}") for k in keys]
        colors = [LABEL_COLORS.get(k, "#999") for k in keys]
        counts = [label_counts.get(k, 0) for k in keys]

        n_cls = max(len(keys), 1)
        pad = 40
        plot_w = width - pad - 20
        plot_h = height - pad - 20
        max_count = max(counts) if counts and max(counts) > 0 else 1
        bar_w = plot_w / n_cls * 0.6

        bars = ""
        labels_html = ""
        for i, (cnt, lbl, clr) in enumerate(zip(counts, labels, colors)):
            bx = pad + i * (plot_w / n_cls) + (plot_w / n_cls - bar_w) / 2
            bh = (cnt / max_count) * (plot_h - 20)
            by = pad + plot_h - bh
            bars += (f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
                     f'height="{bh:.1f}" fill="{clr}" rx="3"/>')
            bars += (f'<text x="{bx+bar_w/2:.1f}" y="{max(by-4,0):.1f}" '
                     f'font-size="12" fill="#333" text-anchor="middle">{cnt}</text>')
            labels_html += (f'<text x="{bx+bar_w/2:.1f}" y="{pad+plot_h+14}" '
                            f'font-size="11" fill="#666" text-anchor="middle">{lbl}</text>')

        frame = (f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+plot_h}" stroke="#ccc"/>'
                 f'<line x1="{pad}" y1="{pad+plot_h}" x2="{width-20}" y2="{pad+plot_h}" stroke="#ccc"/>')

        return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                f'{frame}{bars}{labels_html}</svg>')

    def _render_no_data(self, start_ts, end_ts, patient_info) -> str:
        """无数据时生成占位报告"""
        patient = patient_info or {}
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>CO2 呼吸健康监测报告</title></head>
<body>
<h1 style="text-align:center;">CO₂ 呼吸健康监测报告</h1>
<div class="no-data" style="text-align:center;padding:60px;color:#999;">
<p>监测时段 {start_ts[:10]} ~ {end_ts[:10]} 内无数据</p>
<p>患者: {patient.get('name', '—')}</p>
<p style="font-size:12px;color:#ccc;">报告生成时间: {gen_time}</p>
</div>
</body></html>"""
