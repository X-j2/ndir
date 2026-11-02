"""
生成上位机线程架构示意图（论文用 v2）
运行：python gen_thread_arch.py
输出：thread_architecture.png / .svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

FIG_W, FIG_H = 11, 7.6
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.6)
ax.axis('off')

# ── 配色 ──
C_MAIN_BG  = "#EDF4F2"   # 主线程浅绿
C_WORK_BG  = "#E8EDF5"   # 串口线程浅蓝
C_MAIN_ED  = "#2E7D5B"   # 主线程边框绿
C_WORK_ED  = "#3A6EA5"   # 串口边框蓝
C_ACCENT   = "#E67E22"   # AI 推理橙
C_ARROW    = "#333333"
C_SIGNAL   = "#C0392B"   # 跨线程信号红
C_TEXT     = "#222222"
C_LABEL    = "#666666"

def thread_box(x, y, w, h, title, subtitle, facecolor, edgecolor):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.8, zorder=1)
    ax.add_patch(box)
    ax.text(x + 0.2, y + h - 0.22, title, fontsize=13,
            fontweight='bold', color=edgecolor, zorder=3)
    ax.text(x + 0.2, y + h - 0.55, subtitle, fontsize=8.5,
            color=C_LABEL, zorder=3)

def module(x, y, w, h, text, facecolor="#FFFFFF", accent=False,
           fontsize=9.3, ec=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=facecolor,
                         edgecolor=(C_ACCENT if accent else (ec or "#AAAAAA")),
                         linewidth=(2.0 if accent else 1.2), zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=C_TEXT, zorder=4)

def arrow(x1, y1, x2, y2, label=None, color=C_ARROW, lw=1.7,
          style='-|>', lx=0, ly=0.14, fs=8.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=13, color=color, linewidth=lw, zorder=5)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2 + ly, label, fontsize=fs,
                color=C_LABEL, ha='center', zorder=6)

# ═══════════════════ 标题 ═══════════════════
ax.text(FIG_W/2, 7.32, "上位机软件线程架构示意图", ha='center',
        fontsize=17, fontweight='bold', color=C_TEXT)

# ═══════════════════ 主线程容器 ═══════════════════
MAIN_X, MAIN_Y, MAIN_W, MAIN_H = 0.3, 0.7, 10.4, 4.9
thread_box(MAIN_X, MAIN_Y, MAIN_W, MAIN_H,
           "主线程 (GUI 事件循环)",
           "QApplication  ·  QTimer  ·  帧解析  ·  波形绘制  ·  SQLite 写入  ·  AI 推理",
           C_MAIN_BG, C_MAIN_ED)

# ── 主线程：第 1 行（顶部）定时器 ──
module(0.8, 4.95, 2.5, 0.55, "定时器组\nwave 40ms · stats 1s · sim 200ms",
       facecolor=C_MAIN_BG, fontsize=8.6)
# 数据接入
module(4.0, 4.95, 2.6, 0.55, "数据接入  _apply_frame()\n帧解析 + 单位换算 + 分类",
       facecolor=C_MAIN_BG, fontsize=8.6)

# ── 主线程：第 2 行 数据分发 ──
module(0.8, 3.55, 2.5, 0.6, "实时波形\npyqtgraph 滚动绘制", facecolor=C_MAIN_BG)
module(3.9, 3.55, 2.6, 0.6, "仪表盘\nCO2 / 呼吸率 / EtCO2 / 电压", facecolor=C_MAIN_BG, fontsize=8.6)
module(7.0, 3.55, 2.6, 0.6, "SQLite 数据存储\ndevice / alarm / log 三表", facecolor=C_MAIN_BG, fontsize=8.6)

# ── 主线程：第 3 行 AI 与告警 ──
module(0.8, 1.9, 2.5, 0.7, "AI 推理引擎\n每 30 秒一次\nONNX Random Forest 五分类",
       facecolor="#FDF0E0", accent=True, fontsize=8.6)
module(3.9, 1.9, 2.6, 0.7, "告警引擎\n连续异常 + 置信度≥60%\n状态栏闪烁 / 托盘弹窗", facecolor=C_MAIN_BG, fontsize=8.3)
module(7.0, 1.9, 2.6, 0.7, "趋势预警 + 日志\n斜率检测 / EtCO2 基线\n分级日志输出", facecolor=C_MAIN_BG, fontsize=8.3)

# ── 主线程内部箭头（下行数据流）──
arrow(3.3, 5.2, 4.0, 5.2, "原始字节/模拟帧", ly=0.2, lx=0)
arrow(2.0, 4.95, 2.0, 4.15, "采样点", lw=1.2)
arrow(5.3, 4.95, 5.3, 4.15, "浓度+标签", lw=1.2)
arrow(2.05, 3.55, 2.05, 2.6, "30秒窗口", lw=1.2, lx=0)
arrow(5.3, 3.55, 5.3, 2.6, "统计特征", lw=1.2)
arrow(3.3, 2.25, 3.9, 2.25, "诊断结果", lw=1.3)
arrow(6.5, 2.25, 7.0, 2.25, "触发", lw=1.2)
arrow(8.3, 3.55, 8.3, 2.6, "写日志", lw=1.1, lx=0.15)

# ── 数据来源标注（主线程内底部）──
ax.text(0.8, 1.1, "数据来源：STM32 硬件（串口） · 模拟模式（内置波形生成器）",
        fontsize=8.5, color=C_LABEL, style='italic', zorder=3)

# ═══════════════════ 串口 Worker 线程容器 ═══════════════════
WORK_X, WORK_Y, WORK_W, WORK_H = 0.3, -1.15, 10.4, 1.4
thread_box(WORK_X, WORK_Y, WORK_W, WORK_H,
           "串口 Worker 线程 (QThread)",
           "QSerialPort 全部 I/O 在后台执行，避免异常串口阻塞 GUI",
           C_WORK_BG, C_WORK_ED)

# ── 串口线程模块 ──
module(0.8, -0.85, 2.8, 0.6, "QSerialPort\nopen / read / close", facecolor=C_WORK_BG, fontsize=8.6)
module(4.2, -0.85, 2.8, 0.6, "字节流缓冲\n帧同步恢复(64B定长)", facecolor=C_WORK_BG, fontsize=8.6)
module(7.6, -0.85, 2.8, 0.6, "readyRead 回调\nreadAll → data_received", facecolor=C_WORK_BG, fontsize=8.6)

arrow(3.6, -0.55, 4.2, -0.55, "", lw=1.2)
arrow(7.0, -0.55, 7.6, -0.55, "", lw=1.2)

# ═══════════════════ 跨线程信号 ═══════════════════
# 串口线程 → 主线程（数据上送）
arrow(9.0, -0.85, 9.0, 0.7, "", color=C_SIGNAL, lw=2.2)
ax.text(9.15, -0.1, "data_received (bytes)\n[队列信号]", fontsize=8.3,
        color=C_SIGNAL, va='center', zorder=6)
# 主线程 → 串口线程（控制命令）
arrow(9.8, 0.7, 9.8, -0.85, "", color=C_SIGNAL, lw=1.5, style='<|-')
ax.text(10.05, 0.15, "do_open\n/ do_close\n/ do_send", fontsize=7.8,
        color=C_SIGNAL, va='center', zorder=6)

# ── 图例 ──
lg_y = 6.55
ax.text(0.8, lg_y, "图例：", fontsize=9, color=C_TEXT, fontweight='bold')
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((1.4, lg_y - 0.13), 0.34, 0.26,
                       facecolor="#FFFFFF", edgecolor="#AAAAAA", linewidth=1.2))
ax.text(1.9, lg_y, "功能模块", fontsize=8.5, color=C_LABEL, va='center')
ax.add_patch(Rectangle((3.0, lg_y - 0.13), 0.34, 0.26,
                       facecolor="#FDF0E0", edgecolor=C_ACCENT, linewidth=1.8))
ax.text(3.5, lg_y, "AI 推理", fontsize=8.5, color=C_LABEL, va='center')
ax.plot([4.7, 5.3], [lg_y, lg_y], color=C_SIGNAL, lw=2.2)
ax.text(5.5, lg_y, "跨线程信号(队列连接)", fontsize=8.5, color=C_LABEL, va='center')
ax.text(0.8, lg_y - 0.3, "注：跨线程通信均通过 Qt 信号槽队列连接实现，线程安全",
        fontsize=8.5, color=C_LABEL, style='italic')

plt.tight_layout()
fig.savefig("thread_architecture.png", dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig("thread_architecture.svg", bbox_inches='tight', facecolor='white')
print("已生成: thread_architecture.png (300dpi) + .svg")
