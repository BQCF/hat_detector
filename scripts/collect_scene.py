"""现场取证：从摄像头 / 视频 / RTSP 抽帧，攒成自己的场景数据集。

用法：
  python scripts/collect_scene.py --list                       # 看看本机有哪些摄像头
  python scripts/collect_scene.py --source 0 --count 200       # 采 200 张（每个源建议 ≥100）
  python scripts/collect_scene.py --source D:/videos/room.mp4  # 从视频里抽
热键（有预览窗口时）：s 立刻存一张　p 暂停/继续　q 退出

设计要点：
- 按时间间隔抽帧（默认 0.7s），并丢弃「和上一张几乎一样」的画面 —— 否则 200 张里 190 张是空房间。
- 保存为 JPEG，文件名带会话前缀和序号，原始分辨率不缩放。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def open_capture(src: str, width: int, height: int):
    """按字符串判断是摄像头序号还是文件/URL。"""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(src)
    return cap


def list_cameras():
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from core.sources import list_cameras as _lc
        cams = _lc()
    except Exception as e:  # noqa: BLE001
        print(f"枚举失败（{e}），改用序号逐个试")
        return []
    if not cams:
        print("没枚举到摄像头，请用 --source 0 / 1 / 2 试")
        return []
    print("本机摄像头：")
    for idx, label in cams:
        print(f"  --source {idx}   {label}")
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="摄像头序号 / 视频文件 / rtsp 地址")
    ap.add_argument("--out", default="data/scene/raw")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.7, help="每隔多少秒存一张")
    ap.add_argument("--min-diff", type=float, default=5.0,
                    help="与上一张的平均像素差低于此值就跳过（防重复）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--session", default="", help="会话名，默认用时间戳")
    ap.add_argument("--list", action="store_true", help="只列摄像头")
    ap.add_argument("--no-window", action="store_true", help="不弹预览窗口（远程/后台采）")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    session = args.session or datetime.now().strftime("%m%d_%H%M")
    cap = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"打不开采集源：{args.source}（先跑 --list 看序号）")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("源打开了但读不到画面")
    h, w = frame.shape[:2]
    print(f"采集源 {args.source} 已打开：{w}x{h}，目标 {args.count} 张，间隔 {args.interval}s")
    print(f"保存到 {out_dir}（前缀 {session}）")
    if not args.no_window:
        print("热键：s 立刻存一张 / p 暂停 / q 退出")

    prev_gray = None
    saved = skipped = idx = 0
    last_save = 0.0
    paused = False
    force = False
    t0 = time.time()

    while saved < args.count and ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)

        now = time.time()
        due = (now - last_save) >= args.interval
        diff = 999.0 if prev_gray is None else float(np.abs(small - prev_gray).mean())

        if not paused and (force or (due and diff >= args.min_diff)):
            idx += 1
            saved += 1
            name = f"{session}_{saved:04d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            prev_gray = small
            last_save = now
            force = False
            print(f"  [{saved}/{args.count}] {name}  变化度={diff:.1f}", flush=True)
        elif not paused and due and diff < args.min_diff:
            skipped += 1
            last_save = now  # 免得每帧都判一次

        if args.no_window:
            time.sleep(0.01)
            ok, frame = cap.read()
            continue

        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (w, 64), (16, 17, 21), -1)
        cv2.putText(hud, f"saved {saved}/{args.count}   skipped {skipped}   diff {diff:5.1f}",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 232, 235), 2)
        cv2.putText(hud, "move around / change pose   [s]save [p]pause [q]quit",
                    (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1)
        if paused:
            cv2.putText(hud, "PAUSED", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 90, 255), 2)
        cv2.imshow("collect scene", hud)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            force = True
        if key == ord("p"):
            paused = not paused
        ok, frame = cap.read()

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    manifest = {
        "source": args.source, "session": session, "frames": saved,
        "skipped_similar": skipped, "resolution": [w, h],
        "interval": args.interval, "elapsed_sec": round(time.time() - t0, 1),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    mf = out_dir / f"manifest_{session}.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{saved} 张 → {out_dir}（跳过 {skipped} 张几乎一样的画面）")
    if saved < 80:
        print("提示：张数偏少。建议换个时段/换个人/换角度再采一次，凑够 200 张以上效果更稳。")
    print("下一步：python scripts/preannotate.py")


if __name__ == "__main__":
    main()
