#!/usr/bin/env python
# cli.py
# 无界面模式：验证 core 层（检测 / 报警 / 抓拍 / 事件表），也给没有显示器的机器用。
#
# 用法：
#   python cli.py --type image --source data/dataset_2cls/images/val/xxx.jpg
#   python cli.py --type video --source test.mp4 --frames 200
#   python cli.py --type camera --source 0 --show
#   python cli.py --type rtsp  --source rtsp://192.168.1.10:554/stream
#   python cli.py --type image --source a.jpg --no-sound
import argparse
import sys
import time
from pathlib import Path

from core.alarm import AlarmEvent, AlarmManager
from core.config import load_config
from core.detector import Detector
from core.pipeline import InferPipeline
from core.sources import create_source


def on_alarm(ev: AlarmEvent) -> None:
    print(f"  [报警] {ev.time_str}  {ev.cls_name} x{ev.count}  "
          f"conf={ev.max_conf:.3f}  截图={ev.image_path or '（未保存）'}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="安全帽检测 · 无界面模式")
    p.add_argument("--type", default=None, choices=["camera", "video", "image", "rtsp"],
                   help="采集源类型（默认取配置里的 source_type）")
    p.add_argument("--source", default=None,
                   help="摄像头序号 / 文件路径 / rtsp 地址（默认取配置里的 source_value）")
    p.add_argument("--model", default="", help="模型权重路径（默认自动找最新的 best.pt / best.onnx）")
    p.add_argument("--backend", default="", choices=["", "auto", "torch", "onnx"],
                   help="推理后端：torch=ultralytics+GPU，onnx=onnxruntime（默认取配置）")
    p.add_argument("--conf", type=float, default=None, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=None, help="推理分辨率")
    p.add_argument("--frames", type=int, default=0, help="处理多少帧后退出，0=直到采集源结束")
    p.add_argument("--show", action="store_true", help="弹出 OpenCV 窗口实时显示")
    p.add_argument("--no-sound", action="store_true", help="报警时不发声")
    p.add_argument("--config", default="", help="配置文件路径（默认 configs/default.yaml）")
    args = p.parse_args()

    cfg = load_config(args.config or None)
    if args.type:
        cfg.source_type = args.type
    if args.source is not None:
        cfg.source_value = args.source
    if args.conf is not None:
        cfg.conf = args.conf
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.no_sound:
        cfg.alarm.sound = False

    print("=== 安全帽检测 · 无界面模式 ===")
    detector = Detector(model_path=args.model, device=cfg.device, backend=args.backend or cfg.backend,
                        conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, max_det=cfg.max_det)
    print(f"模型  : {detector.model_path}")
    print(f"后端  : {detector.backend_name}   设备: {detector.device}   类别: {list(detector.names.values())}")
    print(f"阈值  : conf={cfg.conf}  imgsz={cfg.imgsz}")

    source = create_source(cfg.source_type, cfg.source_value,
                           cfg.camera_width, cfg.camera_height, cfg.video_loop)
    source.open()
    print(f"采集源: {cfg.source_type} -> {cfg.source_value}")

    alarm = AlarmManager(cfg.alarm, on_alarm=on_alarm)
    pipe = InferPipeline(detector, source, alarm=alarm, dedup=True)
    pipe.start()

    n = 0
    seen_ts = 0.0
    t0 = time.time()
    try:
        while True:
            res = pipe.latest()
            if res is not None and res.timestamp != seen_ts:
                seen_ts = res.timestamp
                n += 1
                status = "报警中" if res.alarming else "正常"
                print(f"[{n:4d}] {res.fps:5.1f} FPS | {res.infer_ms:5.1f} ms | "
                      f"{status} | {res.counts}")
                if args.show:
                    import cv2
                    cv2.imshow("hat_detector (q=quit)", res.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.frames and n >= args.frames:
                    break
            if pipe.error:
                print(f"[错误] {pipe.error}", file=sys.stderr)
                return 1
            if pipe.finished:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n已手动中断。")
    finally:
        pipe.stop()
        if args.show:
            import cv2
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n=== 结束 === 共处理 {n} 帧，用时 {elapsed:.1f}s，"
          f"平均 {n / elapsed if elapsed else 0:.1f} FPS")
    if cfg.alarm.snapshot_dir:
        print(f"抓拍目录: {alarm.snapshot_dir}")
    print(f"事件表  : {alarm.events_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
