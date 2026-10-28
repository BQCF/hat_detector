# 安全帽佩戴检测系统（hat_detector）

基于 YOLOv8 的安全帽佩戴检测：识别画面中的 **未戴安全帽的头部（head）** 与 **安全帽（helmet）**，
提供网页端（上传图片 / 浏览器摄像头）与本地 OpenCV 实时窗口两种使用方式。

![精度与数据](docs/images/01_quality_and_data.png)

## 指标（验证集 883 张 / 3462 个实例）

| 类别 | 训练集实例 | 验证集实例 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| head（未戴安全帽） | 4874 | 896 | 0.842 | 0.974 | **0.972** | 0.675 |
| helmet（安全帽） | 14884 | 2484 | 0.895 | 0.974 | **0.984** | 0.678 |
| person（人） | 473 | 82 | 0.068 | 0.049 | 0.056 | 0.034 |
| **全部** | 20231 | 3462 | 0.602 | 0.666 | 0.671 | 0.462 |

训练配置：`yolov8n` + 50 epochs + `imgsz=640` + `batch=16`，耗时约 1.69 小时（RTX 4060 Laptop 8GB）。

> ⚠️ **关于 person 类**：它只占训练实例的 2.3%，且验证集的 82 个真值只分布在 25 张图里。
> 实测把阈值降到 0.05 也只能检出 25 个、最高置信度 0.287，在默认阈值 0.25 下几乎全部被过滤。
> 也就是说 `mAP50 = 0.671` 这个总分是被 person 类（(0.972+0.984+0.056)/3）拉低的，head/helmet 两类的真实水平是 0.97 / 0.98。
> 下一版将改为 **head/helmet 二分类**（告警逻辑本来就只看 `head`），详见文末「优化路线」。

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

`data.yaml` 已配好 3 个类别与路径。放好后先跑一次数据集体检：

```bash
python scripts/check_dataset.py
python scripts/check_dataset.py --json runs/dataset_check.json   # 顺便落盘一份报告
```

体检内容：图片/标注配对、类别分布、坐标是否越界、框宽高是否为 0、类别 id 是否越界、
图片能否解码、尺寸是否过小、是否有重复图片、train/val 是否混入同一张图。

## 训练

```bash
python train.py                 # 默认 50 轮，输出到 runs/detect/hat_detector/
python train.py --epochs 100
python train.py --data coco8.yaml   # 用官方小数据集先跑通流程
```

> 注意：`train.py` 目前用了 `exist_ok=True`，同名实验目录会被**覆盖**，需要对比实验时请先备份 `runs/`。

## 评估

```bash
python evaluate.py              # 在验证集上评估 best.pt
```

输出 mAP@0.5、mAP@0.5:0.95、Precision、Recall。测试集（883 张 / 3346 个实例）目前还没有接入评估流程。

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
├─ train.py               # YOLOv8 微调训练
├─ evaluate.py            # 验证集指标评估
├─ data.yaml              # 数据集配置（3 类：head / helmet / person）
├─ scripts/
│  └─ check_dataset.py    # 数据集体检
├─ docs/images/           # README 用图
├─ requirements.txt       # 运行依赖
├─ requirements-train.txt # 训练 / GPU 依赖
├─ data/                  # 数据集（不入库）
└─ runs/                  # 训练与评估产物、模型权重（不入库）
```

## 常见问题

**Q：模型文件在哪？**
`runs/detect/hat_detector/weights/best.pt`（约 6.2MB，3.0M 参数）。仓库不提供权重，需自行训练，或把权重放进上述路径。

**Q：报 `找不到训练好的模型`？**
先 `python train.py` 完成训练，或从别处拷贝 `best.pt` 到 `runs/detect/hat_detector/weights/`。

**Q：torch 报 CUDA 不可用？**
说明装的是 CPU 版 torch：`pip uninstall torch torchvision` 后按 `requirements-train.txt` 安装 CUDA 版。

**Q：网页版摄像头黑屏？**
必须是 `http://127.0.0.1:7860` 访问（浏览器只允许安全上下文使用摄像头），且不能同时被其他程序占用。

## 已知限制

- person 类基本不可用（见上文），界面里的 person 计数不可信；
- 网页版实时链路约 5 FPS；
- 报警只在本地窗口版的 HUD 与终端提示，没有声音、抓拍留证与事件记录；
- 没有桌面程序，必须开浏览器 + 端口访问；
- 测试集尚未接入评估。

## 优化路线

1. **模型**：改为 head/helmet 二分类重训，补测试集评估，训练脚本加实验命名与增强参数；
2. **桌面程序**：PySide6 三线程管道（采集 / 推理 / UI），支持摄像头、视频、图片、RTSP，报警做闭环（连续帧去抖 + 声音 + 抓拍 + CSV 事件）；
3. **部署**：导出 ONNX 用 onnxruntime 推理，PyInstaller 打包成双击可用的 exe（目标 ≤120MB，免 Python 环境）；
4. **增强**：ROI 区域、ByteTrack 去重、多路摄像头、事件导出 Excel。

## 数据与致谢

- 检测框架：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- 数据集：Hard Hat Workers 系列（Roboflow 导出，3 类标注）。使用/分发前请确认原始数据集的许可条款。
