"""
Excel 数据导出模块 — 生成带图表的 .xlsx 文件
    from excel_exporter import ExcelExporter
    ExcelExporter.export(rows, output_path)
"""
import os

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter

# 标签映射（与 co2_monitor.py 一致，v2.2：4 类，键 0/1/2/3）
LABEL_NAMES = {0: "正常", 1: "急促", 2: "屏息", 3: "异常"}

HEADER_FILL = PatternFill(start_color="DDEBDD", end_color="DDEBDD", fill_type="solid")
HEADER_FONT = Font(bold=True)


class ExcelExporter:
    """将设备采样数据导出为带趋势图表的 Excel 文件"""

    @staticmethod
    def export(rows: list, output_path: str) -> str:
        """导出数据到 Excel
        Args:
            rows: query_device_data 返回的记录列表
            output_path: 输出文件路径
        Returns:
            输出文件路径
        """
        output_path = os.path.abspath(output_path)
        wb = Workbook()

        # ── 表1：数据明细 ──
        ws = wb.active
        ws.title = "数据明细"
        headers = ["时间", "CO2(mmHg)", "呼吸率", "EtCO2(mmHg)", "AI标签", "置信度"]
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = HEADER_FONT
            cell.fill = HEADER_FILL
            cell.alignment = Alignment(horizontal="center")
        # 冻结首行
        ws.freeze_panes = "A2"

        for r, row_data in enumerate(rows, 2):
            ts = (row_data.get('timestamp') or '').replace('T', ' ')[:19]
            label = LABEL_NAMES.get(row_data.get('ai_label'), "") if row_data.get('ai_label') is not None else ""
            conf = row_data.get('confidence')
            ws.cell(row=r, column=1, value=ts)
            ws.cell(row=r, column=2, value=round(row_data.get('co2'), 1) if row_data.get('co2') is not None else None)
            ws.cell(row=r, column=3, value=round(row_data.get('rr'), 1) if row_data.get('rr') is not None else None)
            ws.cell(row=r, column=4, value=round(row_data.get('etco2'), 1) if row_data.get('etco2') is not None else None)
            ws.cell(row=r, column=5, value=label)
            ws.cell(row=r, column=6, value=round(conf, 3) if conf is not None else None)

        # 列宽自适应
        for col in range(1, 7):
            ws.column_dimensions[get_column_letter(col)].width = 14

        # ── 表2：趋势图表 ──
        ws_chart = wb.create_sheet("趋势图表")

        # 呼吸率趋势折线图
        if len(rows) >= 2:
            chart = LineChart()
            chart.title = "呼吸率趋势"
            chart.style = 12
            chart.y_axis.title = "次/分"
            chart.x_axis.title = "记录序号"
            # 数据引用（跳过表头，取呼吸率列）
            data = Reference(ws, min_col=3, min_row=1, max_row=min(len(rows) + 1, 2000))
            cats = Reference(ws, min_col=1, min_row=2, max_row=min(len(rows) + 1, 2000))
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.height = 10
            chart.width = 25
            ws_chart.add_chart(chart, "B2")

            # EtCO2 趋势折线图
            chart2 = LineChart()
            chart2.title = "EtCO2 趋势"
            chart2.style = 13
            chart2.y_axis.title = "mmHg"
            chart2.x_axis.title = "记录序号"
            data2 = Reference(ws, min_col=4, min_row=1, max_row=min(len(rows) + 1, 2000))
            chart2.add_data(data2, titles_from_data=True)
            chart2.set_categories(cats)
            chart2.height = 10
            chart2.width = 25
            ws_chart.add_chart(chart2, "B18")

            # CO2 浓度趋势折线图
            chart3 = LineChart()
            chart3.title = "CO2 浓度趋势"
            chart3.style = 14
            chart3.y_axis.title = "mmHg"
            chart3.x_axis.title = "记录序号"
            data3 = Reference(ws, min_col=2, min_row=1, max_row=min(len(rows) + 1, 2000))
            chart3.add_data(data3, titles_from_data=True)
            chart3.set_categories(cats)
            chart3.height = 10
            chart3.width = 25
            ws_chart.add_chart(chart3, "B34")

        wb.save(output_path)
        return output_path
