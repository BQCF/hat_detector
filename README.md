# 安全帽佩戴检测系统（hat_detector）

基于 YOLOv8 的安全帽佩戴检测：识别画面中的 **未戴安全帽的头部（head）** 与 **安全帽（helmet）**，
提供网页端（上传图片 / 浏览器摄像头）与本地 OpenCV 实时窗口两种使用方式。

![精度与数据](docs/images/01_quality_and_data.png)

## 指标

当前模型是 **head / helmet 二分类**（`yolov8n` 微调 50 轮，`imgsz=640`，`batch=16`，耗时约 1.6 小时，RTX 4060 Laptop 8GB）：

| 类别 | 训练集实例 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| head（未戴安全帽） | 4874 | 0.9304 | 0.9554 | **0.9709** | 0.6764 |
| helmet（安全帽） | 14884 | 0.9642 | 0.9428 | **0.9842** | 0.6823 |
| **总体** | 19758 | 0.9473 | 0.9491 | **0.9775** | 0.6793 |

*（val 集 883 张 / 3380 个实例）*

**测试集**（883 张 / 3286 个实例）：head 0.9637 / helmet 0.9765 / **总体 mAP50 0.9701**（mAP50-95 0.6692），
与 val 只差 **0.74 个百分点**，泛化正常。

![二分类重训对比](docs/images/03_2cls_vs_3cls.png)

> 📄 完整对比（含被替换掉的旧 3 类模型）见 [docs/report_2cls.md](docs/report_2cls.md)。
>
> **为什么不是 3 类（head / helmet / person）**：person 全集只有 615 个实例（占 2.3%），
> 旧模型在测试集上对它的**召回率是 0**（60 个真值一个都没检出），把总体 mAP50 从 0.977 拉到 0.66。
> 而 head / helmet 两类的水平重训前后没变——去 person 不是为了变强，而是为了让指标说实话。
> 告警逻辑本来就只看 `head`，二分类才是这个项目该有的形态。

## 环境要求

- Windows / Linux，Python 3.10+
- 已验证组合：Windows 11 · Python 3.12.10 · ultralytics 8.4.142 · torch 2.6.0+cu124 · gradio 6.26.0 · opencv-python 5.0.0.93 · RTX 4060 Laptop 8GB
- 无 GPU 也能跑（CPU 推理，速度较慢）

## 安装

```bash
# 1) 创建虚拟环境（可选，推荐）
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2) 装依赖（CPU 或按 ultralytics 默认选择 torch）
pip install -r requirements.txt

# 需要 GPU 训练时，先装 CUDA 版 torch，再装其余
pip install -r requirements-train.txt
pip install -r requirements.txt
```

## 数据准备

数据集体积约 259MB，**不随仓库提供**（含第三方数据许可限制）。请按下面的结构放置：

```
data/dataset/
├─ images/{train,val,test}/   # 图片
└─ labels/{train,val,test}/   # 与图片同名的 .txt 标注
```

`data.yaml` 是 3 类配置（head / helmet / person），`data2cls.yaml` 是当前使用的二分类配置。
放好数据后先体检一次：

```bash
python scripts/check_dataset.py                      # 体检 3 类数据集
python scripts/check_dataset.py --data data2cls.yaml # 体检二分类数据集
```

体检内容：图片/标注配对、类别分布、坐标是否越界、框宽高是否为 0、类别 id 是否越界、
图片能否解码、尺寸是否过小、是否有重复图片、train/val 是否混入同一张图。

### 派生二分类数据集

只有 3 类数据集时，一条命令就能派生出 head/helmet 二分类版本（图片走硬链接，不额外占磁盘）：

```bash
python scripts/make_2class.py              # 生成 data/dataset_2cls/ + data2cls.yaml
python scripts/make_2class.py --copy       # 改为复制图片
python scripts/make_2class.py --keep head  # 只保留某一类
```

## 训练

实验目录自动带时间戳（`runs/detect/<名称>_<YYYYmmdd_HHMMSS>/`），**不会覆盖历史结果**：

```bash
python train.py --data data2cls.yaml --name hat_2cls   # 当前推荐：二分类
python train.py                                        # 用 data.yaml 训练 3 类
python train.py --epochs 100 --cache ram               # 多轮 + 图片缓存提速
python train.py --model yolov11n.pt --cos-lr           # 换模型 / 余弦退火学习率
python train.py --data coco8.yaml --no-check           # 官方小数据集先跑通流程
```

## 评估

```bash
python evaluate.py --split val                                # 验证集（自动取最新的 best.pt）
python evaluate.py --split test                               # 测试集
python evaluate.py --data data2cls.yaml --split test --json runs/eval_test.json
python evaluate.py --model runs/detect/xxx/weights/best.pt    # 指定模型
```

输出分类指标表（Precision / Recall / mAP50 / mAP50-95），`--json` 可落盘。

## 使用

### 网页版（推荐演示用）

```bash
python app.py                                  # 默认 http://127.0.0.1:7860
APP_PORT=8080 python app.py                     # 换端口
```

- 「图片检测」页：上传图片 → 点「开始检测」→ 出带框结果与分类统计；
- 「实时检测」页：先点「▶ 开始检测」再授权摄像头；「⏹ 停止检测」后只回显画面、不再占 GPU；
- 局域网访问：`server_name` 已设为 `0.0.0.0`，用 `http://<本机IP>:7860` 即可。**但浏览器摄像头只在 `http://127.0.0.1` 下可用**，局域网设备请用「图片检测」。

> 实时检测链路约 5 FPS：瓶颈不是模型（GPU 单帧仅 7.4ms / 135 FPS），
> 而是浏览器摄像头 → WebSocket → 服务端 → 回传图片这一来一回的传输开销（`stream_every=0.2`）。
> 想要真正的实时体验，用下面的本地窗口模式。

### 本地实时窗口

```bash
python demo_opencv.py                                   # 默认摄像头 0
python demo_opencv.py --source video.mp4 --tracker       # 视频 + ByteTrack 追踪
python demo_opencv.py --conf 0.3 --imgsz 416             # 调阈值与推理分辨率
python demo_opencv.py --source 0 --device cpu            # 强制 CPU
```

窗口内按键：`q` 退出、`s` 保存当前帧、空格暂停/继续。

```bash
python demo.py --source photo.jpg --save    # 单张图片检测并保存
python demo.py --source 0 --conf 0.5        # 轻量版实时检测
```

## 目录结构

```
hat_detector/
├─ app.py                 # Gradio 网页版（图片检测 + 浏览器摄像头实时检测）
├─ demo.py                # 轻量命令行/实时演示（超简单版）
├─ demo_opencv.py         # OpenCV 实时窗口（异步读帧 + FPS/HUD + 连续帧报警）
├─ train.py               # YOLOv8 微调训练（实验目录带时间戳，不覆盖历史结果）
├─ evaluate.py            # val / test 指标评估，可落盘 JSON
├─ data.yaml              # 3 类数据集配置（head / helmet / person）
├─ data2cls.yaml          # 二分类数据集配置（head / helmet，当前使用）
├─ scripts/
│  ├─ check_dataset.py    # 数据集体检
│  └─ make_2class.py      # 从 3 类派生二分类数据集
├─ docs/
│  ├─ report_2cls.md      # 二分类重训对比报告
│  └─ images/             # README 与报告用图
├─ requirements.txt       # 运行依赖
├─ requirements-train.txt # 训练 / GPU 依赖
├─ data/                  # 数据集（不入库）
└─ runs/                  # 训练与评估产物、模型权重（不入库）
```

## 常见问题

**Q：模型文件在哪？**
每次训练会在 `runs/detect/<名称>_<时间戳>/weights/best.pt` 生成一份（约 6.2MB，3.0M 参数）。
仓库不提供权重，需自行训练；`evaluate.py` 不指定 `--model` 时会自动取最新的那份。

**Q：报 `找不到训练好的模型`？**
先 `python train.py --data data2cls.yaml --name hat_2cls` 完成训练。

**Q：torch 报 CUDA 不可用？**
说明装的是 CPU 版 torch：`pip uninstall torch torchvision` 后按 `requirements-train.txt` 安装 CUDA 版。

**Q：网页版摄像头黑屏？**
必须是 `http://127.0.0.1:7860` 访问（浏览器只允许安全上下文使用摄像头），且不能同时被其他程序占用。

## 已知限制

- 网页版实时链路约 5 FPS；
- 报警只在本地窗口版的 HUD 与终端提示，没有声音、抓拍留证与事件记录；
- 没有桌面程序，必须开浏览器 + 端口访问。

## 优化路线

1. ✅ **模型**（已完成）：head/helmet 二分类重训、测试集接入评估、训练脚本带时间戳实验目录 —— 见 [docs/report_2cls.md](docs/report_2cls.md)；
2. **桌面程序**：PySide6 三线程管道（采集 / 推理 / UI），支持摄像头、视频、图片、RTSP，报警做闭环（连续帧去抖 + 声音 + 抓拍 + CSV 事件）；
3. **部署**：导出 ONNX 用 onnxruntime 推理，PyInstaller 打包成双击可用的 exe（目标 ≤120MB，免 Python 环境）；
4. **增强**：ROI 区域、ByteTrack 去重、多路摄像头、事件导出 Excel。

## 数据与致谢

- 检测框架：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- 数据集：Hard Hat Workers 系列（Roboflow 导出，3 类标注）。使用/分发前请确认原始数据集的许可条款。
