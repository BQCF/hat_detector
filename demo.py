# demo.py
# 实时推理演示：摄像头 / 图片 / 视频 检测；检测到"未戴安全帽的头部(head)"时控制台报警
# 数据集 3 类：head(未戴帽头部) / helmet(安全帽) / person(人)
# 用法：
#   python demo.py                              # 摄像头实时检测（默认源 0）
#   python demo.py --source video.mp4           # 视频文件
#   python demo.py --source photo.jpg --save    # 单张图片并保存标注结果
#   python demo.py --source 0 --conf 0.5        # 调置信度阈值
# 提示：窗口显示时按 q 退出
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# 项目根目录（本文件所在目录），保证从任意目录运行都指向正确的文件
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "runs" / "detect" / "hat_detector" / "weights"
BEST_MODEL = WEIGHTS_DIR / "best.pt"     # 训练产出的最佳模型
PRETRAINED = BASE_DIR / "yolov8n.pt"     # 官方预训练模型（首次使用自动下载，约 6MB）
OUT_DIR = BASE_DIR / "runs" / "detect" / "demo"   # --save 时标注结果输出目录
ALARM_CLASSES = {"head"}  # 需要报警的类别：head=未戴帽的头部（本数据集 3 类 head/helmet/person）


def load_model() -> tuple[YOLO, bool]:
    """优先加载训练好的 best.pt；不存在则退回官方预训练模型（仅用于演示流程）。"""
    if BEST_MODEL.exists():
        return YOLO(BEST_MODEL), True
    print(f"[提示] 未找到训练模型：{BEST_MODEL}")
    print("       改用官方预训练模型演示（yolov8n.pt 是 COCO 80 类，检测不到安全帽/背心）。")
    print("       训练完成后再次运行即可使用 best.pt。")
    if not PRETRAINED.exists():
        print("[提示] yolov8n.pt 尚不存在，首次运行会自动下载（约 6MB）。")
    return YOLO(PRETRAINED), False


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 安全帽检测（head/helmet/person，摄像头、图片或视频）")
    parser.add_argument("--source", type=str, default="0", help="检测源：0=摄像头，或图片/视频路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (0,1]，默认 0.5")
    parser.add_argument("--save", action="store_true", help="保存标注结果到 runs/detect/demo/（默认不保存）")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        raise SystemExit("[错误] --conf 必须是 (0,1] 之间的数")

    model, is_trained = load_model()
    alarm_active = is_trained and ALARM_CLASSES.issubset(set(model.names.values()))
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"开始检测，源: {args.source}    设备: {device}")
    print("窗口显示时按 q 键退出。")

    results = model.predict(
        source=args.source,             # "0"=摄像头，也可以是图片/视频路径
        conf=args.conf,                 # 置信度阈值
        imgsz=640,
        device=device,
        show=True,                      # 实时显示窗口
        save=args.save,                 # 是否保存标注结果
        project=str(OUT_DIR.parent),    # --save 时结果存到 runs/detect/demo/
        name=OUT_DIR.name,
        exist_ok=True,
        stream=True,                    # 流式处理（摄像头/视频更流畅）
        verbose=False,                  # 关闭逐帧刷屏，改由下方按秒汇总
    )

    frames = 0
    last_stat = time.time()
    last_alert = 0.0
    try:
        for r in results:
            frames += 1

            # 报警：检测到"未戴帽 / 未穿背心"（同一类别 1 秒最多提示一次，避免刷屏）
            if alarm_active and r.boxes is not None and len(r.boxes) > 0:
                now = time.time()
                alarm = sorted({model.names[int(c)] for c in r.boxes.cls.tolist()} & ALARM_CLASSES)
                if alarm and now - last_alert >= 1.0:
                    print(f"[警告] {time.strftime('%H:%M:%S')} 检测到：{'、'.join(alarm)}", flush=True)
                    last_alert = now

            # 每秒打印一次实时 FPS
            now = time.time()
            if now - last_stat >= 1.0:
                print(f"FPS: {frames / (now - last_stat):.1f}", flush=True)
                frames, last_stat = 0, now

            # q 键退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n已手动中断。")
    finally:
        cv2.destroyAllWindows()

    if args.save:
        print(f"标注结果已保存到：{OUT_DIR}")


if __name__ == "__main__":
    main()
