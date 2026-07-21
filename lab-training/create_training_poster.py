from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1440, 3600
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "ci_lab_training_rollup_poster.png"

NAVY = "#083344"
TEAL = "#0F766E"
CYAN = "#0891B2"
SKY = "#E0F2FE"
MINT = "#CCFBF1"
AMBER = "#F59E0B"
AMBER_LIGHT = "#FEF3C7"
INK = "#0F172A"
MUTED = "#475569"
WHITE = "#FFFFFF"
PALE = "#F8FAFC"


def font(size, bold=False):
    family = "msyhbd.ttc" if bold else "msyh.ttc"
    path = Path("C:/Windows/Fonts") / family
    return ImageFont.truetype(str(path), size)


def rounded(draw, box, fill, radius=28, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, fill=INK, bold=False, spacing=10, anchor=None):
    draw.multiline_text(
        xy,
        value,
        font=font(size, bold),
        fill=fill,
        spacing=spacing,
        anchor=anchor,
    )


def center_text(draw, x, y, value, size, fill=INK, bold=False):
    text(draw, (x, y), value, size, fill, bold, anchor="ma")


def draw_wave(draw, left, top, width, height, color):
    points = []
    for x in range(width + 1):
        y = top + height / 2 + (height * 0.38) * __import__("math").sin(x / 44)
        points.append((left + x, int(y)))
    draw.line(points, fill=color, width=7)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), PALE)
    draw = ImageDraw.Draw(image)

    # Top field
    draw.rectangle((0, 0, WIDTH, 890), fill=NAVY)
    draw.rectangle((0, 810, WIDTH, 890), fill=TEAL)
    draw.ellipse((1030, 70, 1450, 490), fill="#0F4C5C")
    draw.ellipse((1110, 150, 1370, 410), outline="#34D399", width=7)
    draw.arc((1045, 85, 1435, 475), 30, 320, fill="#67E8F9", width=7)
    draw_wave(draw, 950, 610, 420, 120, "#67E8F9")

    rounded(draw, (92, 92, 407, 158), "#164E63", radius=22)
    center_text(draw, 250, 125, "N3lab | 联合实验室", 27, "#BAE6FD", True)
    text(draw, (92, 228), "CI / 助听器实验室", 62, "#A7F3D0", True)
    text(draw, (88, 312), "开发技能培训", 122, WHITE, True)
    text(draw, (96, 480), "从 Python 到端到端 CI 语音处理 Pipeline", 39, "#E0F2FE", False)

    rounded(draw, (92, 615, 842, 752), "#0F4C5C", radius=28)
    center_text(draw, 250, 664, "7 日密集培训", 35, WHITE, True)
    draw.line((438, 642, 438, 725), fill="#67E8F9", width=3)
    center_text(draw, 555, 664, "每天一个上午", 33, WHITE, True)
    draw.line((713, 642, 713, 725), fill="#67E8F9", width=3)
    center_text(draw, 778, 664, "3 小时 / 天", 31, WHITE, True)

    # Introduction
    rounded(draw, (72, 950, 1368, 1255), WHITE, radius=32)
    text(draw, (116, 1002), "培训简介", 42, TEAL, True)
    text(
        draw,
        (116, 1076),
        "面向 CI / 助听器研究实验室成员，以真实声学与听觉研究任务为主线，\n"
        "完成 Python 开发、Linux 环境、深度学习、音频分类、DeepACE、\n"
        "DeepFilterNet、ASR 与端到端 Pipeline 的系统训练。",
        35,
        INK,
        False,
        18,
    )

    # Content cards
    text(draw, (88, 1345), "培训内容", 48, NAVY, True)
    content = [
        ("01", "Python 与工程基础", "信号处理、面向对象、调试、Git", CYAN),
        ("02", "Linux 与深度学习环境", "SSH、Conda、PyTorch、Jupyter", TEAL),
        ("03", "深度学习与音频分类", "MLP、CNN、CRNN、VAD", "#2563EB"),
        ("04", "CI 语音处理模型", "ACE、DeepACE、DeepFilterNet", "#7C3AED"),
        ("05", "语音识别与系统整合", "Whisper、GET 声码器、Pipeline", AMBER),
    ]
    card_y = 1435
    for index, (number, title, description, color) in enumerate(content):
        y = card_y + index * 145
        rounded(draw, (82, y, 1358, y + 112), WHITE, radius=24)
        rounded(draw, (102, y + 18, 190, y + 94), color, radius=20)
        center_text(draw, 146, y + 56, number, 28, WHITE, True)
        text(draw, (220, y + 22), title, 31, INK, True)
        text(draw, (620, y + 28), description, 26, MUTED)

    # Schedule
    schedule_top = 2220
    rounded(draw, (72, schedule_top, 1368, 3168), WHITE, radius=34)
    text(draw, (110, schedule_top + 45), "7 天培训安排", 46, NAVY, True)
    rounded(draw, (106, schedule_top + 125, 1334, schedule_top + 198), "#E2E8F0", radius=15)
    text(draw, (140, schedule_top + 144), "日期", 25, MUTED, True)
    text(draw, (355, schedule_top + 144), "上午课程", 25, MUTED, True)
    text(draw, (924, schedule_top + 144), "核心主题", 25, MUTED, True)

    days = [
        ("第 1 天", "模块 0  Python 编程基础", "语法 / OOP / 调试 / Git"),
        ("第 2 天", "模块 1  Linux 与环境搭建", "Linux / Conda / PyTorch / Jupyter"),
        ("第 3 天", "模块 2  深度学习入门", "MLP / CNN / 训练技巧"),
        ("第 4 天", "模块 3  音频分类", "特征 / CRNN / CI 分类任务"),
        ("第 5 天", "模块 4  DeepACE 模型解析", "ACE / 论文 / 代码 / 修改假设"),
        ("第 6 天", "模块 5  DeepFilterNet 模型解析", "语音增强 / ERB / CI 整合"),
        ("第 7 天", "模块 6  ASR 与端到端 Pipeline", "Whisper / GET / 端到端评估"),
    ]
    row_height = 116
    for i, (day, course, topic) in enumerate(days):
        y = schedule_top + 216 + i * row_height
        fill = "#F0FDFA" if i % 2 == 0 else "#F8FAFC"
        rounded(draw, (106, y, 1334, y + 92), fill, radius=15)
        center_text(draw, 190, y + 46, day, 27, TEAL, True)
        text(draw, (355, y + 24), course, 28, INK, True)
        text(draw, (924, y + 28), topic, 24, MUTED)

    # Venue and host
    venue_top = 3240
    rounded(draw, (72, venue_top, 1368, 3538), NAVY, radius=34)
    text(draw, (112, venue_top + 40), "培训地点", 40, "#A7F3D0", True)
    text(
        draw,
        (112, venue_top + 104),
        "华南理工大学五山校区 23 号楼\n深圳龙岗医院联合实验室\n线上会议同步进行",
        32,
        WHITE,
        False,
        14,
    )
    draw.line((764, venue_top + 42, 764, venue_top + 250), fill="#2DD4BF", width=3)
    text(draw, (810, venue_top + 40), "主办方", 40, "#A7F3D0", True)
    text(
        draw,
        (810, venue_top + 104),
        "华南理工大学物理与光电学院 N3lab\n深圳龙岗区耳鼻咽喉医院-\n华南理工大学联合实验室",
        28,
        WHITE,
        False,
        14,
    )

    text(draw, (WIDTH // 2, 3562), "CI / 助听器研究实验室开发技能培训", 24, MUTED, True, anchor="ma")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
