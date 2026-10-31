# 现场数据标注指南

> 目标：把 `data/scene/labels/{train,val,test}/*.txt` 里模型预标注的框改正。
> 你只需要**删错框、补漏框、改类别**，不用从零画。全部标完大约 150 张只需 1 小时。

---

## 1. 先跑完前两步

```bash
python scripts/collect_scene.py --list                        # 看摄像头序号
python scripts/collect_scene.py --source 0 --count 200        # 采 200 张（摆不同姿态/角度）
python scripts/preannotate.py                                 # 预标注 + 切分 train/val/test
```

采的时候有讲究，直接决定模型能学到多少：

| 要采到的情况 | 为什么 |
| --- | --- |
| 有人戴、有人没戴，同一画面里都有 | 这是唯一能让模型分清两类的对照 |
| 站远（3–5 米）、站近（1 米）各来一些 | 补小目标（<0.5% 面积漏检率 41%） |
| 走动中的动态模糊 | 实测运动模糊让召回掉到 0.70，必须让它见过 |
| 逆光/窗边、晚上开灯 | 逆光掉 12pp |
| 空镜（没人）20–30 张 | 降"凭空多框"的误报 |
| 侧面、背面、低头看手机 | 工地数据集里几乎没有这些姿态 |

---

## 2. 标注工具

### 方案 A：labelImg（最省事）

```bash
python -m pip install labelImg -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m labelImg
```

打开后按这个顺序设置（**顺序错了框会存错地方**）：

1. 左侧 `Format` 选 **YOLO**
2. `Open Dir` → `D:\projects\hat_detector\data\scene\images\train`
3. `Change Save Dir` → `D:\projects\hat_detector\data\scene\labels\train`（预标注的 txt 就在这里，打开后应该能看到已有的框）
4. 类别人工确认是 `head`、`helmet` 两个（`data/scene/classes.txt` 里有）

跑 `val` / `test` 两个子集时重复 2–3 两步。

### 方案 B：X-AnyLabeling（有 AI 辅助，界面更现代）

```bash
python -m pip install anylabeling -i https://pypi.tuna.tsinghua.edu.cn/simple
```
加载模型后可直接在界面上自动框，缺点是要自己导出成 YOLO 格式。

> 如果两个都不想装，跟我说，我可以在这个项目里加一个内置标注器（PySide6，跟主程序同一套暗色主题，双击就能用）。

---

## 3. 标什么、不标什么（统一口径，别犹豫）

| 情况 | 标成 | 说明 |
| --- | --- | --- |
| 头上戴着安全帽 | `helmet` | 框紧贴帽子，能带一点头也行，但别框到肩膀 |
| 光头/头发露在外面 | `head` | 框住整个头部，含头发 |
| 戴棒球帽、毛线帽、鸭舌帽 | `head` | **只有安全帽才是 helmet**，这是业务定义 |
| 帽子拿在手上 / 挂在脖子上 | 不标 | 没戴在头上就不算 |
| 被遮挡 50% 以上 | 不标 | 被遮一半以内照常标 |
| 小于 8×8 像素、糊成一团认不出 | 不标 | 这段模型学不会，硬标只会教坏它 |
| 海报、电视、手机屏幕里的头 | 不标 | 这一条专治"凭空多框" |
| 水印、台标、logo | 不标 | 原数据集里就有人把台标标成 helmet，别再放进自己的数据 |
| 背影只露后脑勺 | 标 `head`（若可见） | 这是现场最常见、原数据集最缺的姿态 |

快捷键：`W` 画框 · `A/D` 上一张/下一张 · `Del` 删框 · `Ctrl+S` 保存 · `Ctrl+U` 批量载入目录。

---

## 4. 标完怎么验收

```bash
# ① 现场测试集上的真实水平（这才是优化前的正确起点）
python evaluate.py --data data/scene/data.yaml --split test --json runs/eval_scene_before.json

# ② 错在哪（同样的 7 张图，换成你的场景）
python scripts/error_analysis.py --data data/scene/data.yaml --split test

# ③ 抽查自己标得对不对（拼图）
python scripts/fn_label_check.py
```

拿到 ①② 之后，再决定是走"数据补强 + 重训"还是"先改后处理"。**没这组数字之前的任何调参都是猜。**

---

## 5. 数据量建议

| 场景 | 建议张数 | 说明 |
| --- | --- | --- |
| 只想先看有没有救 | 50 张 | 只能定性看 |
| 有效微调 | 150–300 张 | **推荐**，通常能把现场召回拉 10pp 以上 |
| 想要稳定 | 500+ 张，覆盖不同时段/人 | 到这个量级就值得每周做增量训练了 |
