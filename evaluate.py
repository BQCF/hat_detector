# evaluate.py
# 评估脚本：在验证集上评测训练好的最佳模型，并打印关键指标
# 用法：激活虚拟环境后，在 hat_detector 目录下执行：
#   python evaluate.py                     # 用本项目 data.yaml（需数据集就绪）
#   python evaluate.py --data coco8.yaml   # 先跑通流程（配合 coco8 冒烟训练结果）
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# 项目根目录（本文件所在目录），保证从任意目录运行都指向正确的文件
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data.yaml"                                  # 数据集配置文件
WEIGHTS_DIR = BASE_DIR / "runs" / "detect" / "hat_detector" / "weights"
BEST_MODEL = WEIGHTS_DIR / "best.pt"                                # 训练产出的最佳模型


def find_best_model() -> Path:
    """返回 best.pt 路径；不存在时给出明确报错与补救提示。"""
    if BEST_MODEL.exists():
        return BEST_MODEL
    print(f"[错误] 找不到模型权重：{BEST_MODEL}")
    print("请先运行 python train.py 完成训练。")
    last_model = WEIGHTS_DIR / "last.pt"
    if last_model.exists():
        print(f"提示：检测到训练中断留下的 last.pt（{last_model}），")
        print("      如需继续，可在 train.py 中给 model.train(...) 加 resume=True 参数。")
    raise SystemExit(1)


def main() -> None:
    # 0. 命令行参数：--data 指定评估用的数据集配置
    parser = argparse.ArgumentParser(description="在验证集上评估训练好的模型")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集配置文件（默认 data.yaml；跑通流程可用官方自带 coco8.yaml）")
    args = parser.parse_args()
    data_yaml = Path(args.data) if args.data else DATA_YAML

    # 1. 找到训练好的模型
    model_path = find_best_model()
    device = "0" if torch.cuda.is_available() else "cpu"

    # 2. 用本项目 data.yaml 评估前先确认验证集存在（--data 指定其它数据集时不检查）
    if args.data is None:
        val_dir = BASE_DIR / "data" / "dataset" / "images" / "val"
        if not val_dir.is_dir():
            print(f"[错误] 验证集不存在：{val_dir}")
            print("请先放入数据集；或先用官方小数据集验证流程：")
            print("  python evaluate.py --data coco8.yaml")
            raise SystemExit(1)

    # 3. 加载模型并在验证集上评估（验证集由 data.yaml 的 val 路径指定）
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device)

    # 4. 打印关键指标（面试/复试前背住这几个数字）
    print("\n" + "=" * 50)
    print(f"评估结果（验证集）模型: {model_path.name}")
    print("=" * 50)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
