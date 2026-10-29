# 阶段 1 · 二分类重训对比报告

> 日期：2026-10-29　｜　环境：Windows 11 · Python 3.12.10 · torch 2.6.0+cu124 · ultralytics 8.4.142 · RTX 4060 Laptop 8GB
> 本次改动：**只把 person 类从训练与指标里去掉**，其余超参数与旧模型完全一致（`yolov8n` / 50 epochs / `imgsz=640` / `batch=16` / `patience=20`）。

![二分类重训对比](images/03_2cls_vs_3cls.png)

## 1. 结论先行

**这次重训没有让模型变强，而是让指标说实话。**

head / helmet 两类的检测水平，重训前后几乎一模一样；变化的只是把那个根本学不出来的 `person` 类从平均分里摘掉，
总体 mAP50 因此从 0.66 回到 0.97 —— 这才是模型的真实水平。

| | 旧·3 类模型 | 新·2 类模型 | 变化 |
| --- | --- | --- | --- |
| head（test mAP50） | 0.9639 | 0.9637 | **-0.0002（没变）** |
| helmet（test mAP50） | 0.9840 | 0.9765 | -0.0075（波动内） |
| person（test mAP50） | 0.0296 | — | 已移除 |
| **总体（test mAP50）** | 0.6592 | **0.9701** | **+0.31** |
| **总体（test mAP50-95）** | 0.4506 | **0.6692** | **+0.22** |

## 2. 四格完整对比

同一套评估代码（`evaluate.py`）、同一批图片，分别跑 val 与 test：

| 模型 | 集 | 图片 | head mAP50 | helmet mAP50 | person mAP50 | 总体 mAP50 | 总体 mAP50-95 | P | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 旧·3 类 | val | 883 | 0.9722 | 0.9832 | 0.0598 | 0.6717 | 0.4624 | 0.6071 | 0.6699 |
| 旧·3 类 | test | 883 | 0.9639 | 0.9840 | 0.0296 | 0.6592 | 0.4506 | 0.9613 | 0.6271 |
| 新·2 类 | val | 883 | 0.9709 | 0.9842 | — | **0.9775** | **0.6793** | 0.9473 | 0.9491 |
| 新·2 类 | test | 883 | 0.9637 | 0.9765 | — | **0.9701** | **0.6692** | 0.9484 | 0.9403 |

**泛化检查**：新模型 val 0.9775 → test 0.9701，只掉 **0.74 个百分点**（验收线是 2%），没有过拟合。

## 3. person 类为什么必须去掉

用旧模型在测试集上的完整分类指标：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| head | 0.9296 | 0.9321 | 0.9639 | 0.6608 |
| helmet | 0.9543 | 0.9491 | 0.9840 | 0.6727 |
| **person** | 1.0000 | **0.0000** | 0.0296 | 0.0183 |

- **召回率 0**：测试集里 60 个 person 真值，一个都没检出。Precision 显示 1.0000 是因为"没有预测就不存在误报"，纯属虚高。
- 根因是数据：person 全集只有 473（train）+ 82（val）+ 60（test）= **615 个实例**，占全部实例的 2.3%，
  且验证/测试集里分散在 25 张图中。这类量级靠调参救不回来。
- 业务上也没有价值：告警规则是 `ALARM_CLASSES = {"head"}`，只看"有没有人没戴帽子"，从不需要 person。

## 4. 本次交付

| 文件 | 作用 |
| --- | --- |
| `scripts/make_2class.py` | 从 3 类数据集派生 head/helmet 二分类数据集（图片走硬链接，不占额外磁盘） |
| `data/dataset_2cls/` | 派生出的二分类数据集：5269 / 883 / 883 张图，实例 19758 / 3380 / 3286 |
| `data2cls.yaml` | 二分类数据集配置（`nc=2`） |
| `train.py` | 实验目录自动带时间戳 `<前缀>_<YYYYmmdd_HHMMSS>`，`exist_ok=False`，不再覆盖历史结果；新增 `--cache` / `--cos-lr` / `--close-mosaic` / `--seed` / `--name` |
| `evaluate.py` | 新增 `--split val|test`、`--model`、`--json`、`--conf`；打印分类指标表并落盘 JSON |
| `runs/detect/hat_2cls_20261029_085425/` | 本次训练产物（时间戳目录，旧的 `hat_detector/` 完整保留） |
| `runs/eval_{val,test}_{2cls,3cls}.json` | 四格对比的原始数据 |

## 5. 验收对照

| 验收项 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- |
| head/helmet 每类 mAP50 | ≥ 0.95 | 0.9709 / 0.9842 | ✅ |
| head/helmet 每类 Recall | ≥ 0.93 | 0.9554 / 0.9428 | ✅ |
| test 与 val 指标差距 | < 2% | 0.74% | ✅ |
| 时间戳实验目录不覆盖历史 | — | `hat_2cls_20261029_085425` 新建，`hat_detector/` 未动 | ✅ |
| 测试集接入评估 | — | `evaluate.py --split test` 可用，已产出 JSON | ✅ |

## 6. 复现方式

```bash
python scripts/make_2class.py                                    # 派生二分类数据集（幂等）
python scripts/check_dataset.py --data data2cls.yaml             # 体检（当前 0 错 0 警）
python train.py --data data2cls.yaml --name hat_2cls --epochs 50 # 训练，实验目录带时间戳
python evaluate.py --data data2cls.yaml --split val  --json runs/eval_val_2cls.json
python evaluate.py --data data2cls.yaml --split test --json runs/eval_test_2cls.json
```

## 7. 下一步

1. **桌面程序**：PySide6 三线程管道，支持摄像头/视频/图片/RTSP，报警闭环（连续帧去抖 + 声音 + 抓拍 + CSV 事件）；
2. **部署**：`best.pt` 导出 ONNX，onnxruntime 推理，PyInstaller 打包成免 Python 环境的 exe；
3. **增强**：ROI 区域、ByteTrack 去重、多路摄像头、事件导出 Excel。
