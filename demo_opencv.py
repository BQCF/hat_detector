# demo_opencv.py
# 用法:
#   python demo_opencv.py                 # 默认摄像头 0
#   python demo_opencv.py --source 0
#   python demo_opencv.py --source video.mp4 --tracker
#   python demo_opencv.py --conf 0.3 --imgsz 416
# 按 q 退出，按 s 保存当前帧，按空格暂停/继续

import argparse
from pathlib import Path
import queue
import threading
import time

import cv2
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL = BASE_DIR / "runs" / "detect" / "hat_detector" / "weights" / "best.pt"
PRETRAINED = BASE_DIR / "yolov8n.pt"
ALARM_CLASSES = {"head"}


class FrameReader:
    """异步视频/摄像头读取类，防止主线程因采集阻塞"""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开数据源: {source}")

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            if self.q.full():
                try:
                    self.q.get_nowait()  # 扔掉旧帧，保证实时性
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return True, self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        self.stopped = True
        self.cap.release()


def load_model():
    if BEST_MODEL.exists():
        print(f"[OK] 加载训练模型: {BEST_MODEL}")
        return YOLO(str(BEST_MODEL)), True
    print("[WARN] 未找到 best.pt，使用预训练模型演示")
    return YOLO(str(PRETRAINED)), False


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 实时安全帽检测（性能优化版）")
    p.add_argument(
        "--source", type=str, default="0", help="0=摄像头，或视频/图片路径"
    )
    p.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    p.add_argument(
        "--imgsz", type=int, default=416, help="推理分辨率，416 更快，640 更准"
    )
    p.add_argument(
        "--device", type=str, default="", help="空=自动，0=GPU，cpu=CPU"
    )
    p.add_argument(
        "--tracker", action="store_true", help="是否启用 ByteTrack 目标追踪"
    )
    return p.parse_args()


def main():
    args = parse_args()
    model, is_trained = load_model()

    # 设备判断与 Half 设置
    if args.device:
        device = args.device
    else:
        device = "0" if torch.cuda.is_available() else "cpu"

    use_half = device != "cpu" and torch.cuda.is_available()

    source = int(args.source) if args.source.isdigit() else args.source
    reader = FrameReader(source)

    win = "PPE Detection (q=Quit, s=Save, Space=Pause)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    frames = 0
    t_fps = time.time()
    fps_show = 0.0
    last_alert = 0.0

    # 连续帧告警确认队列，降低误报率
    alarm_counter = 0
    ALARM_THRESHOLD = 3  # 连续 3 帧确认后才报警

    print(
        f"设备={device} quantize={'fp16' if use_half else 'fp32'} imgsz={args.imgsz} conf={args.conf} tracker={args.tracker}"
    )

    while True:
        if not paused:
            ok, frame = reader.read()
            if not ok or frame is None:
                print("视频读取结束或超时")
                break

            t0 = time.time()

            # 推理模式：支持 ByteTrack 追踪或纯推理
            predict_kwargs = dict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                quantize=16 if use_half else None,  # 16=FP16 半精度，None=FP32（ultralytics 已废弃 half）
                verbose=False,
                max_det=20,
            )

            if args.tracker:
                results = model.track(
                    **predict_kwargs, tracker="bytetrack.yaml", persist=True
                )
            else:
                results = model.predict(**predict_kwargs)

            infer_ms = (time.time() - t0) * 1000
            result = results[0]
            plotted = result.plot(line_width=2)

            # 类别统计与报警分析
            names = result.names
            counts = {}
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.int().cpu().tolist()
                for c in classes:
                    name = names[c]
                    counts[name] = counts.get(name, 0) + 1

            raw_alarm = bool(
                is_trained and sorted(set(counts) & ALARM_CLASSES)
            )

            # 防误报逻辑：连续累计确认
            if raw_alarm:
                alarm_counter += 1
            else:
                alarm_counter = max(0, alarm_counter - 1)

            is_alarming = alarm_counter >= ALARM_THRESHOLD
            now = time.time()

            if is_alarming and now - last_alert >= 1.0:
                print(
                    f"[警告] {time.strftime('%H:%M:%S')} 确认违规: 未佩戴安全帽!"
                )
                last_alert = now

            # 性能面板绘制
            frames += 1
            if now - t_fps >= 1.0:
                fps_show = frames / (now - t_fps)
                frames = 0
                t_fps = now

            status = "ALARM: NO HELMET" if is_alarming else "OK"
            color = (0, 0, 255) if is_alarming else (0, 200, 0)

            # 半透明背景 HUD
            overlay = plotted.copy()
            cv2.rectangle(overlay, (8, 8), (420, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, plotted, 0.4, 0, plotted)

            cv2.putText(
                plotted,
                f"FPS {fps_show:.1f} | Infer: {infer_ms:.0f} ms",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                plotted,
                f"{status} | {counts}",
                (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            cv2.imshow(win, plotted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord("s") and not paused:
            out = BASE_DIR / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(out), plotted)
            print(f"已保存截图: {out}")

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
