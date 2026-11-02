"""
通信驱动模块测试
    python test_serial.py

测试范围：
  - 协议编解码 (encode_frame / decode_frame)
  - 浓度换算 (ppm_to_mmhg / mmhg_to_ppm，v2.3 起取代原电压换算)
  - 采样队列溢出字段 (byte 16-19)
  - 帧同步恢复 (find_frame_sync)
  - 边界值 / 异常输入
  - 可用串口扫描 (list_ports, 需 Qt)
"""
import unittest
import sys
import os

# 加项目目录到 path，以便 import co2_monitor 中的协议函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 导入协议纯函数（不触发 Qt GUI 初始化）──
from co2_monitor import (
    FRAME_SIZE, SAMPLES_PER_FRAME, AI_MAGIC, FRAME_HEADER,
    PPM_TO_MMHG, CONCENTRATION_START,
    AI_FLAG_INDEX, AI_CLASS_INDEX, RR_INDEX, CHECKSUM_INDEX,
    OVERFLOW_INDEX,
    ppm_to_mmhg, mmhg_to_ppm,
    encode_frame, decode_frame, frame_checksum,
    find_frame_sync,
)


class TestOverflowField(unittest.TestCase):
    """byte 16-19：采样队列累计丢弃点数（F407 固件 v2.2 起新增）

    固件把采样放在 TIM3 中断里以精确 25Hz 产生、主循环异步消费。
    队列满时会丢弃采样点并累加此计数，据此可判断波形是否存在缺口。
    """

    def test_overflow_default_zero(self):
        """不指定时默认为 0，且旧固件（此区恒为 0）解析结果一致"""
        frame = encode_frame([100, 200, 300, 400, 500], ai_class=0)
        result = decode_frame(frame)
        self.assertEqual(result['overflow'], 0)

    def test_overflow_roundtrip(self):
        """任意值往返一致"""
        for value in [1, 255, 256, 65535, 65536, 12345678, 0xFFFFFFFF]:
            frame = encode_frame([0] * 5, overflow=value)
            self.assertEqual(decode_frame(frame)['overflow'], value,
                             f"溢出值 {value} 往返不一致")

    def test_overflow_big_endian(self):
        """大端序：0x01020304 → byte16..19 = 01 02 03 04"""
        frame = encode_frame([0] * 5, overflow=0x01020304)
        self.assertEqual(list(frame[OVERFLOW_INDEX:OVERFLOW_INDEX + 4]),
                         [0x01, 0x02, 0x03, 0x04])

    def test_overflow_index_is_16(self):
        """字段位置固定在 byte 16（与固件 usart.c 一致）"""
        self.assertEqual(OVERFLOW_INDEX, 16)

    def test_checksum_excludes_overflow(self):
        """校验和只覆盖 byte 0-14，不含溢出字段"""
        frame_a = encode_frame([100] * 5, overflow=0)
        frame_b = encode_frame([100] * 5, overflow=999999)
        self.assertEqual(frame_a[CHECKSUM_INDEX], frame_b[CHECKSUM_INDEX])

    def test_overflow_negative_clamped(self):
        """负值被夹到 0，不产生非法字节"""
        frame = encode_frame([0] * 5, overflow=-5)
        self.assertEqual(decode_frame(frame)['overflow'], 0)

    def test_overflow_does_not_affect_other_fields(self):
        """加了溢出字段后，浓度/AI/呼吸率解析均不受影响"""
        frame = encode_frame([100, 200, 300, 400, 500],
                             ai_class=3, rr_per_min=18.5, overflow=777)
        result = decode_frame(frame)
        self.assertEqual(result['ai_class'], 3)
        self.assertAlmostEqual(result['rr_value'], 18.5, places=5)
        self.assertTrue(result['ai_valid'])
        self.assertAlmostEqual(result['concentration_ppm'][0], 100.0, places=3)
        self.assertEqual(result['overflow'], 777)


class TestPpmMmhgConversion(unittest.TestCase):
    """PPM ←→ mmHg 换算测试（v2.3 起取代原电压换算）"""

    def test_ppm_to_mmhg_basic(self):
        """基本换算：约 50000 ppm 对应 38 mmHg（呼气末典型值）"""
        self.assertAlmostEqual(ppm_to_mmhg(50000.0), 38.0, places=4)

    def test_ppm_to_mmhg_factor(self):
        """换算因子为 760/1e6"""
        self.assertAlmostEqual(PPM_TO_MMHG, 760.0 / 1e6, places=10)
        self.assertAlmostEqual(ppm_to_mmhg(1.0), PPM_TO_MMHG, places=10)

    def test_mmhg_to_ppm_basic(self):
        """反向换算"""
        self.assertAlmostEqual(mmhg_to_ppm(38.0), 50000.0, places=4)

    def test_roundtrip(self):
        """往返换算不丢失精度"""
        for original in (0.0, 450.0, 5000.0, 25000.0, 50000.0):
            back = ppm_to_mmhg(mmhg_to_ppm(original))
            self.assertAlmostEqual(original, back, places=6,
                                   msg=f"{original} ppm 往返不一致")

    def test_zero(self):
        """零值"""
        self.assertAlmostEqual(ppm_to_mmhg(0.0), 0.0, places=6)
        self.assertAlmostEqual(mmhg_to_ppm(0.0), 0.0, places=6)

    def test_air_level(self):
        """空气本底：450 ppm ≈ 0.34 mmHg"""
        self.assertAlmostEqual(ppm_to_mmhg(450.0), 0.342, places=3)

    def test_consistent_with_firmware(self):
        """与固件 ai_bridge.c 的 AB_PPM_TO_MMHG_SCALE 一致"""
        self.assertAlmostEqual(PPM_TO_MMHG, 0.00076, places=10)


class TestEncodeDecodeFrame(unittest.TestCase):
    """帧编解码测试"""

    def test_encode_frame_size(self):
        """编码输出必须是 64 字节"""
        voltages = [500.0, 600.0, 700.0, 800.0, 900.0]
        frame = encode_frame(voltages)
        self.assertEqual(len(frame), FRAME_SIZE)

    def test_encode_without_ai(self):
        """无 AI 标签的编码"""
        voltages = [100.0] * 5
        frame = encode_frame(voltages)
        self.assertEqual(frame[0], FRAME_HEADER)   # 帧头
        self.assertEqual(frame[AI_FLAG_INDEX], 0x00)   # AI 标记位未设置
        self.assertEqual(frame[AI_CLASS_INDEX], 0x00)

    def test_encode_with_ai(self):
        """含 AI 标签的编码"""
        voltages = [300.0] * 5
        frame = encode_frame(voltages, ai_class=0, rr_per_min=14.0)
        self.assertEqual(frame[0], FRAME_HEADER)
        self.assertEqual(frame[AI_FLAG_INDEX], AI_MAGIC)
        self.assertEqual(frame[AI_CLASS_INDEX], 0)  # class 0

        rr_x10 = (frame[RR_INDEX] << 8) | frame[RR_INDEX + 1]
        self.assertEqual(rr_x10, 140)   # 14.0 × 10
        # 校验和正确
        self.assertEqual(frame[CHECKSUM_INDEX], frame_checksum(bytes(frame)))

    def test_roundtrip_no_ai(self):
        """编码→解码往返（无 AI），浓度为 ppm 整数"""
        concentrations = [100.0, 200.0, 300.0, 400.0, 500.0]
        frame = encode_frame(concentrations)
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        for j in range(5):
            self.assertAlmostEqual(result['concentration_ppm'][j],
                                   concentrations[j], places=3)
        self.assertFalse(result['ai_valid'])

    def test_roundtrip_with_ai(self):
        """编码→解码往返（含 AI）"""
        voltages = [250.0, 350.0, 450.0, 550.0, 650.0]
        frame = encode_frame(voltages, ai_class=2, rr_per_min=5.5)
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        self.assertTrue(result['ai_valid'])
        self.assertEqual(result['ai_class'], 2)
        self.assertAlmostEqual(result['rr_value'], 5.5, places=0)

    def test_all_four_classes(self):
        """全部 4 种分类（0/1/2/3）编码后解码正确"""
        for cls in (0, 1, 2, 3):
            frame = encode_frame([500.0] * 5, ai_class=cls, rr_per_min=12.0)
            result = decode_frame(frame)
            self.assertEqual(result['ai_class'], cls,
                             f"class {cls} 编解码不一致")


class TestDecodeEdgeCases(unittest.TestCase):
    """解码异常输入测试"""

    def test_decode_empty(self):
        """空字节 → None"""
        self.assertIsNone(decode_frame(b''))

    def test_decode_too_short(self):
        """不足 64 字节 → None"""
        self.assertIsNone(decode_frame(b'\x00' * 10))

    def test_decode_exact_size(self):
        """刚好 64 字节但有帧头无AI → 可解码"""
        data = bytearray(FRAME_SIZE)
        data[0] = FRAME_HEADER
        data[CHECKSUM_INDEX] = frame_checksum(bytes(data))  # 头+0 = 0xFF 校验
        frame = decode_frame(bytes(data))
        self.assertIsNotNone(frame)
        self.assertFalse(frame['ai_valid'])

    def test_decode_larger_than_frame(self):
        """超过 64 字节 → 只解析前 64 字节"""
        data = bytearray(100)
        data[0] = FRAME_HEADER
        data[CHECKSUM_INDEX] = frame_checksum(bytes(data[:FRAME_SIZE]))
        frame = decode_frame(bytes(data))
        self.assertIsNotNone(frame)

    def test_bad_checksum_rejected(self):
        """校验和不符 → None（防错位垃圾被当真）"""
        data = bytearray(FRAME_SIZE)
        data[0] = FRAME_HEADER
        data[CHECKSUM_INDEX] = (frame_checksum(bytes(data)) + 1) & 0xFF  # 故意弄错
        self.assertIsNone(decode_frame(bytes(data)))

    def test_bad_header_rejected(self):
        """帧头不是 0xFF → None"""
        data = bytearray(FRAME_SIZE)
        data[0] = 0x00
        data[CHECKSUM_INDEX] = frame_checksum(bytes(data))
        self.assertIsNone(decode_frame(bytes(data)))

    def test_magic_byte_position(self):
        """AI_MAGIC 出现在正确位置 (byte 11)"""
        data = bytearray(FRAME_SIZE)
        data[0] = FRAME_HEADER
        data[AI_FLAG_INDEX] = AI_MAGIC
        data[AI_CLASS_INDEX] = 3
        data[RR_INDEX] = 0x00
        data[RR_INDEX + 1] = 0xA0  # 160 → 16.0 次/分
        data[CHECKSUM_INDEX] = frame_checksum(bytes(data))
        frame = decode_frame(bytes(data))
        self.assertTrue(frame['ai_valid'])
        self.assertEqual(frame['ai_class'], 3)
        self.assertAlmostEqual(frame['rr_value'], 16.0, places=0)

    def test_magic_missing(self):
        """未设置 AI_MAGIC → ai_valid = False"""
        data = bytearray(FRAME_SIZE)
        data[0] = FRAME_HEADER
        data[AI_CLASS_INDEX] = 0  # 即使 class 设置了，magic 不在也是无效
        data[CHECKSUM_INDEX] = frame_checksum(bytes(data))
        frame = decode_frame(bytes(data))
        self.assertFalse(frame['ai_valid'])


class TestBoundaryValues(unittest.TestCase):
    """边界值测试"""

    def test_min_concentration(self):
        """最小浓度 0 ppm"""
        concentrations = [0.0] * 5
        frame = encode_frame(concentrations)
        result = decode_frame(frame)
        for v in result['concentration_ppm']:
            self.assertAlmostEqual(v, 0.0, places=3)

    def test_max_concentration(self):
        """uint16 上限 65535 ppm（固件实际夹在 50000，此处测字段本身的容量）"""
        concentrations = [65535.0] * 5
        frame = encode_frame(concentrations)
        self.assertEqual(len(frame), FRAME_SIZE)
        for j in range(5):
            ppm = (frame[CONCENTRATION_START + j * 2] << 8) | \
                  frame[CONCENTRATION_START + j * 2 + 1]
            self.assertEqual(ppm, 65535)

    def test_firmware_max_ppm(self):
        """固件实际最大浓度 50000 ppm 能正确编码且不溢出"""
        frame = encode_frame([50000.0] * 5)
        result = decode_frame(frame)
        for v in result['concentration_ppm']:
            self.assertEqual(v, 50000.0)

    def test_concentration_clamp_negative(self):
        """负浓度被钳位到 0"""
        concentrations = [-1000.0, 0.0, 0.0, 0.0, 0.0]
        frame = encode_frame(concentrations)
        ppm0 = (frame[CONCENTRATION_START] << 8) | frame[CONCENTRATION_START + 1]
        self.assertEqual(ppm0, 0)

    def test_concentration_clamp_overflow(self):
        """超范围浓度被钳位到 65535"""
        concentrations = [99999.0, 0.0, 0.0, 0.0, 0.0]
        frame = encode_frame(concentrations)
        ppm0 = (frame[CONCENTRATION_START] << 8) | frame[CONCENTRATION_START + 1]
        self.assertEqual(ppm0, 65535)

    def test_rr_overflow(self):
        """呼吸率超 6553.5 被钳位"""
        frame = encode_frame([500.0] * 5, ai_class=0, rr_per_min=9999.0)
        rr_x10 = (frame[RR_INDEX] << 8) | frame[RR_INDEX + 1]
        self.assertEqual(rr_x10, 65535)

    def test_rr_zero(self):
        """RR = 0"""
        frame = encode_frame([500.0] * 5, ai_class=0, rr_per_min=0.0)
        rr_x10 = (frame[RR_INDEX] << 8) | frame[RR_INDEX + 1]
        self.assertEqual(rr_x10, 0)


class TestFrameSync(unittest.TestCase):
    """帧同步恢复测试"""

    def test_aligned_buffer(self):
        """已对齐的缓冲区 → offset=0"""
        voltages = [300.0] * 5
        frame = encode_frame(voltages, ai_class=1, rr_per_min=20.0)
        offset = find_frame_sync(frame)
        self.assertEqual(offset, 0)

    def test_misaligned_buffer(self):
        """前面塞垃圾字节（含 0xFF 但校验不过）→ 应找到真实帧偏移"""
        voltages = [300.0] * 5
        frame = encode_frame(voltages, ai_class=0, rr_per_min=14.0)
        # 前面塞 3 字节非 0xFF 垃圾，偏移 3 字节
        shifted = b'\x11\x22\x33' + frame
        padded = shifted + b'\x00' * max(0, FRAME_SIZE * 2 - len(shifted))
        offset = find_frame_sync(padded)
        self.assertEqual(offset, 3,
                         f"预期偏移 3，实际 {offset}")

    def test_garbage_with_false_header(self):
        """垃圾里含假 0xFF（校验不过）→ 跳过它找到真帧"""
        voltages = [300.0] * 5
        frame = encode_frame(voltages, ai_class=0, rr_per_min=14.0)
        # 前面塞 0xFF + 随机字节，0xFF 处校验不过，真帧在偏移 4
        shifted = b'\xFF\x11\x22\x33' + frame
        padded = shifted + b'\x00' * max(0, FRAME_SIZE * 2 - len(shifted))
        offset = find_frame_sync(padded)
        self.assertEqual(offset, 4,
                         f"预期跳过假帧头到偏移 4，实际 {offset}")

    def test_no_valid_frame_in_buffer(self):
        """缓冲区中无有效帧（无 0xFF 或无校验通过）→ -1"""
        data = b'\x00' * 100
        offset = find_frame_sync(data)
        self.assertEqual(offset, -1)
        # 只有假 0xFF、无完整帧 → -1
        fake = b'\xFF' + b'\x00' * 70
        self.assertEqual(find_frame_sync(fake), -1)

    def test_short_buffer(self):
        """过短缓冲区 → -1"""
        self.assertEqual(find_frame_sync(b'\xFF'), -1)
        self.assertEqual(find_frame_sync(b''), -1)

    def test_header_in_middle_of_stream(self):
        """完整帧嵌入流中间 → 返回正确偏移"""
        voltages = [300.0] * 5
        frame = encode_frame(voltages, ai_class=0, rr_per_min=14.0)
        # 前面 15 字节垃圾，真帧在偏移 15
        data = b'\xAB' * 15 + frame
        padded = data + b'\x00' * max(0, FRAME_SIZE * 2 - len(data))
        offset = find_frame_sync(padded)
        self.assertEqual(offset, 15)


class TestAvailablePorts(unittest.TestCase):
    """可用串口扫描测试（需 Qt）"""

    def test_list_ports_no_crash(self):
        """列出可用串口不应抛异常"""
        try:
            from PySide6.QtSerialPort import QSerialPortInfo
            ports = []
            for info in QSerialPortInfo.availablePorts():
                if info.portName():
                    ports.append(info.portName())
            # 无论有没有串口，都不应崩溃
            self.assertIsInstance(ports, list)
        except ImportError:
            self.skipTest("PySide6 未安装，跳过串口列表测试")


class TestFrameConstants(unittest.TestCase):
    """协议常量一致性测试"""

    def test_frame_size(self):
        self.assertEqual(FRAME_SIZE, 64)

    def test_samples_per_frame(self):
        self.assertEqual(SAMPLES_PER_FRAME, 5)

    def test_frame_header(self):
        self.assertEqual(FRAME_HEADER, 0xFF)

    def test_ai_magic(self):
        self.assertEqual(AI_MAGIC, 0xAA)

    def test_conversion_consistent(self):
        """换算因子与固件一致，且正反互为逆运算"""
        self.assertAlmostEqual(ppm_to_mmhg(1000.0), PPM_TO_MMHG * 1000.0, places=8)
        self.assertAlmostEqual(
            ppm_to_mmhg(mmhg_to_ppm(12345.0)), 12345.0, places=6)


class TestSimDataPipeline(unittest.TestCase):
    """模拟数据 → 编码 → 解码 → 应用 全链路测试"""

    def test_simulate_ten_frames(self):
        """模拟 10 帧数据：encode → decode 全链路不抛异常"""
        import random
        random.seed(42)
        for _ in range(10):
            # 浓度范围覆盖空气本底(数百 ppm)到呼气末(数万 ppm)
            concentrations = [random.uniform(0, 50000) for _ in range(5)]
            ai_class = random.choice([0, 1, 2, 3])
            rr = random.uniform(0, 40)
            # 编码
            frame = encode_frame(concentrations, ai_class, rr)
            self.assertEqual(len(frame), FRAME_SIZE)
            # 解码
            result = decode_frame(frame)
            self.assertIsNotNone(result)
            self.assertTrue(result['ai_valid'])
            self.assertEqual(result['ai_class'], ai_class)
            # 浓度值在容差范围内（编码有 int() 截断，容忍 1 ppm）
            for j in range(5):
                self.assertAlmostEqual(result['concentration_ppm'][j],
                                       concentrations[j], delta=1.0)


if __name__ == '__main__':
    print("=" * 60)
    print("通信驱动模块测试")
    print(f"帧大小: {FRAME_SIZE}  每帧采样: {SAMPLES_PER_FRAME}  帧头: 0x{FRAME_HEADER:02X}")
    print(f"浓度起始: byte{CONCENTRATION_START}  AI标志: 0x{AI_MAGIC:02X}@byte{AI_FLAG_INDEX}  校验和: byte{CHECKSUM_INDEX}")
    print(f"浓度换算: ppm×{PPM_TO_MMHG} = mmHg  (v2.3 起字节1-10为 ppm)")
    print("=" * 60)
    unittest.main(verbosity=2)
