# app.py
# Gradio 网页检测界面：安全帽佩戴检测系统（上传图片检测 / 浏览器摄像头实时检测）
# 用法：激活虚拟环境后，在 hat_detector 目录下执行  python app.py
#      浏览器会自动打开 http://127.0.0.1:7860（局域网设备可访问 http://<本机IP>:7860）
# 提示：浏览器摄像头仅在 http://127.0.0.1 下可用（浏览器安全限制），局域网访问请用"图片检测"
import os
import socket
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

# 项目根目录（本文件所在目录），保证从任意目录运行都指向正确的文件
BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL = BASE_DIR / "runs" / "detect" / "hat_detector" / "weights" / "best.pt"

# 三类（必须与 data.yaml 一致）与配色：head=红(未戴帽/危险) helmet=蓝 person=绿
CLASS_COLORS = {"head": "#FF4D4F", "helmet": "#1890FF", "person": "#52C41A"}

# 全局加载一次模型，界面所有功能共用
if not BEST_MODEL.exists():
    print(f"[错误] 找不到训练好的模型：{BEST_MODEL}")
    print("请先运行 python train.py 完成训练。")
    raise SystemExit(1)
model = YOLO(BEST_MODEL)
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# 实时检测性能参数（图片检测仍走 640 高质量）
WEB_IMGSZ = 320           # 摄像头推理分辨率：越低越快，320 兼顾速度与精度
WEB_MAX_SIDE = 640        # 摄像头帧长边上限：超过则等比缩小，降低画幅与回传开销
WEB_MAX_DET = 20          # 每帧最多保留的检测框数
WEB_STREAM_EVERY = 0.2    # 摄像头抓取间隔（秒）≈ 5 帧/秒
WEB_HALF = DEVICE == "0"  # GPU 用 FP16 半精度加速


def infer(img_rgb: np.ndarray, conf: float, imgsz: int = 640, max_det: int = 300) -> tuple[np.ndarray, dict, float]:
    """对一张图做检测。返回 (带框标注图 RGB, 统计 dict, 耗时 ms)。

    Gradio 传来的图是 RGB，而 ultralytics/cv2 按 BGR 处理才能让标注框配色正确。
    imgsz 越小推理越快（图片检测用 640，摄像头用 320）。
    """
    img = img_rgb
    if img.ndim == 2:                              # 灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                        # 带透明通道的 RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:                                          # 常规 RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model.predict(source=img, conf=conf, imgsz=imgsz, device=DEVICE,
                            quantize=16 if WEB_HALF else None, max_det=max_det, verbose=False)
    res = results[0]
    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)  # plot() 返回 BGR，转回 RGB 显示

    counts = {name: 0 for name in model.names.values()}
    confs: list[float] = []
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for c in boxes.cls.tolist():
            counts[model.names[int(c)]] += 1
        confs = boxes.conf.tolist()
    stats = {
        **counts,
        "total": sum(counts.values()),
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
    }
    ms = float(sum(res.speed.values())) if hasattr(res, "speed") else 0.0
    return annotated, stats, ms


def stats_html(stats: dict, ms: float, source: str) -> str:
    """生成"统计表格 + 报警横幅"的 HTML 片段。"""
    n_head = stats["head"]
    if n_head > 0:
        banner = (f'<div style="background:#7a1f1f;color:#ffd9d9;padding:14px 18px;border-radius:10px;'
                  f'border:1px solid #ff4d4f;font-size:18px;font-weight:700;">'
                  f'⚠ 报警：检测到 {n_head} 处未佩戴安全帽（head）</div>')
    else:
        banner = ('<div style="background:#123f24;color:#c6f6d5;padding:14px 18px;border-radius:10px;'
                  'border:1px solid #52c41a;font-size:18px;font-weight:700;">'
                  '✓ 未发现未佩戴安全帽，状态安全</div>')

    rows = ""
    for name in model.names.values():
        color = CLASS_COLORS.get(name, "#ffffff")
        rows += (f'<tr><td style="padding:6px 12px;"><span style="display:inline-block;width:12px;height:12px;'
                 f'background:{color};border-radius:3px;margin-right:8px;"></span>{name}</td>'
                 f'<td style="padding:6px 12px;text-align:right;font-weight:600;">{stats[name]}</td></tr>')
    table = (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 16px;">'
             f'<div style="color:#8b949e;font-size:13px;margin-bottom:6px;">{source} · 推理耗时 {ms:.0f} ms · '
             f'置信度阈值生效中</div>'
             f'<table style="width:100%;color:#e6edf3;border-collapse:collapse;">{rows}'
             f'<tr style="border-top:1px solid #30363d;"><td style="padding:6px 12px;font-weight:700;">总计</td>'
             f'<td style="padding:6px 12px;text-align:right;font-weight:700;">{stats["total"]}</td></tr>'
             f'<tr><td style="padding:6px 12px;color:#8b949e;">平均置信度</td>'
             f'<td style="padding:6px 12px;text-align:right;color:#8b949e;">{stats["avg_conf"]:.3f}</td></tr>'
             f'</table></div>')
    return banner + '<div style="height:12px;"></div>' + table


def fn_image(image: np.ndarray | None, conf: float) -> tuple[np.ndarray | None, str]:
    """Tab1 图片检测：按钮点击时触发。"""
    if image is None:
        return None, '<div style="color:#8b949e;">请先上传一张图片。</div>'
    annotated, stats, ms = infer(image, conf)
    return annotated, stats_html(stats, ms, "图片检测")


def running_html() -> str:
    return ('<div style="color:#3fb950;font-weight:700;padding:8px 2px;font-size:16px;">'
            '🟢 检测运行中</div>')


def stopped_html() -> str:
    return ('<div style="color:#f85149;font-weight:700;padding:8px 2px;font-size:16px;">'
            '🔴 检测已停止</div>')


def start_detection(is_running: bool) -> tuple[bool, str]:
    """「开始检测」按钮：置状态为运行中。"""
    return True, running_html()


def stop_detection(is_running: bool) -> tuple[bool, str]:
    """「停止检测」按钮：置状态为停止。"""
    return False, stopped_html()


def downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    """长边超过 max_side 时等比缩小，降低画幅与回传开销。"""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


_last_webcam_html = ""
_last_webcam_html_t = 0.0


def webcam_html(stats: dict, ms: float) -> str:
    """实时检测的统计/报警 HTML，1 秒最多重建一次，避免每帧拼接字符串。"""
    global _last_webcam_html, _last_webcam_html_t
    now = time.time()
    if now - _last_webcam_html_t >= 1.0:
        _last_webcam_html = stats_html(stats, ms, "实时摄像头")
        _last_webcam_html_t = now
    return _last_webcam_html


_webcam_frames = 0


def fn_webcam(frame, conf: float, is_running: bool):
    """Tab2 实时检测：浏览器摄像头流式输入，每帧调用一次。

    返回 (带框图, 状态说明 HTML)。is_running=False 时不做检测、直接原样回传，
    从而不占用 GPU 算力。
    """
    global _webcam_frames
    _webcam_frames += 1
    if _webcam_frames <= 3:  # 调试：确认帧进入与形状
        print(f"【进入fn_webcam】第{_webcam_frames}帧 type={type(frame).__name__} "
              f"shape={getattr(frame, 'shape', None)} is_running={is_running}", flush=True)
    if frame is None:
        return None, stopped_html()
    if not is_running:
        return downscale(frame, WEB_MAX_SIDE), stopped_html()
    img = downscale(frame, WEB_MAX_SIDE)                     # 减少画幅开销
    annotated, stats, ms = infer(img, conf, imgsz=WEB_IMGSZ, max_det=WEB_MAX_DET)
    return annotated, webcam_html(stats, ms)                 # HTML 1 秒节流重建


def local_ip() -> str:
    """取本机局域网 IP（用于打印给导师访问的地址）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def build_ui() -> gr.Blocks:
    """搭建深色科技风界面。"""
    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_border_color="#30363d",
        block_title_background_fill="#0d1117",
        input_background_fill="#0d1117",
        button_primary_background_fill="#1f6feb",
        button_primary_background_fill_hover="#388bfd",
        border_color_primary="#30363d",
    )
    css = """
    .gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
    #app-header { text-align: center; padding: 10px 0 4px 0; }
    #app-header .sub { color: #8b949e; font-size: 14px; margin-top: 2px; }
    .start-btn { background-color: #1f7d3a !important; }
    .stop-btn  { background-color: #b02418 !important; }
    """
    device_txt = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    header_html = f"""
        <div id="app-header">
          <div style="font-size:26px;font-weight:800;background:linear-gradient(90deg,#58a6ff,#00c6ff);
               -webkit-background-clip:text;color:transparent;">安全帽佩戴检测系统</div>
          <div class="sub">YOLOv8 · 模型 best.pt · 类别 head/helmet/person · 设备 {device_txt}</div>
        </div>
        """

    with gr.Blocks(title="安全帽佩戴检测系统") as demo:
        gr.HTML(header_html)
        conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值（全局生效）")
        is_running = gr.State(value=False)  # 实时检测的"运行/停止"状态

        with gr.Tabs():
            # ---------- Tab1：上传图片检测 ----------
            with gr.Tab("🖼 图片检测"):
                with gr.Row():
                    img_in = gr.Image(type="numpy", sources=["upload"], label="上传图片")
                    img_out = gr.Image(type="numpy", interactive=False, label="检测结果")
                detect_btn = gr.Button("开始检测", variant="primary")
                status_img = gr.HTML()
                detect_btn.click(fn_image, inputs=[img_in, conf_slider], outputs=[img_out, status_img])
                gr.Examples(
                    examples=[str(p) for p in (BASE_DIR / "runs" / "detect" / "demo_val").glob("*.jpg")][:3]
                    if (BASE_DIR / "runs" / "detect" / "demo_val").exists() else [],
                    inputs=img_in,
                    label="示例图片（runs/detect/demo_val 中的验证集推理结果）",
                )

            # ---------- Tab2：实时摄像头检测 ----------
            with gr.Tab("📷 实时检测"):
                with gr.Row():
                    start_btn = gr.Button("▶ 开始检测", variant="primary", elem_classes="start-btn")
                    stop_btn = gr.Button("⏹ 停止检测", elem_classes="stop-btn")
                    state_text = gr.HTML(value=stopped_html())
                with gr.Row():
                    cam_in = gr.Image(sources=["webcam"], type="numpy", label="摄像头画面")
                    cam_out = gr.Image(type="numpy", interactive=False, label="实时检测", streaming=True)
                status_cam = gr.HTML(value=stopped_html())
                # 摄像头流：根据 is_running 状态决定是否真正检测
                cam_in.stream(fn_webcam, inputs=[cam_in, conf_slider, is_running],
                              outputs=[cam_out, status_cam],
                              time_limit=300, stream_every=WEB_STREAM_EVERY, concurrency_limit=30)
                start_btn.click(start_detection, inputs=[is_running], outputs=[is_running, state_text])
                stop_btn.click(stop_detection, inputs=[is_running], outputs=[is_running, state_text])
                gr.HTML(
                    '<div style="color:#8b949e;font-size:13px;padding:6px 2px;">'
                    "操作：先点「▶ 开始检测」再允许摄像头授权；「⏹ 停止检测」后右侧只回显原画面、不再占用 GPU。"
                    "浏览器摄像头仅在 http://127.0.0.1 下可用，局域网访问请用「图片检测」。</div>"
                )

        # 图例
        legend = "".join(
            f'<span style="display:inline-block;width:12px;height:12px;background:{CLASS_COLORS[n]};'
            f'border-radius:3px;margin:0 6px 0 16px;"></span>{n}'
            for n in model.names.values()
        )
        gr.HTML(f'<div style="color:#8b949e;font-size:13px;padding:10px 2px;text-align:center;">'
                f"检测框配色：{legend}　·　模型训练自 Hard Hat Workers 数据集（3 类）</div>")

    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(
        server_name="0.0.0.0",          # 允许局域网访问
        server_port=int(os.environ.get("APP_PORT", "7860")),
        inbrowser=os.environ.get("APP_NO_BROWSER") != "1",
        show_error=True,
        theme=theme,                     # Gradio 6：主题与 CSS 在 launch() 传入
        css=css,
    )
