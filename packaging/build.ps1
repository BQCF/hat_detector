# packaging/build.ps1
# 打包成免 Python 环境的 Windows 桌面程序（onedir）。
#
# 关键取舍：**不带 torch/ultralytics**（加起来 2.5GB），只带 onnxruntime 推理，
# 所以产物能控制在 100MB 级别、冷启动快；代价是纯 CPU 推理（约 7.7ms@320）。
#
# 用法：在项目根目录执行
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# 注意：不能用 $ErrorActionPreference = "Stop" —— PyInstaller 往 stderr 写进度信息，
# PowerShell 会把它当成错误中断脚本；改用 $LASTEXITCODE 判断成功失败。
$ErrorActionPreference = "Continue"

$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
Write-Host "项目目录: $proj"

$py = Join-Path $proj "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { throw "找不到 Python，请先激活虚拟环境" }
}
Write-Host "使用 Python: $py"

if (-not (Test-Path "models\best.onnx")) {
    throw "缺少 models\best.onnx，请先运行  python scripts/export.py"
}

& $py -m PyInstaller --noconfirm --clean `
    --onedir --noconsole `
    --name hat_detector `
    --distpath dist --workpath build --specpath packaging `
    --add-data "$proj\models\best.onnx;models" `
    --exclude-module torch --exclude-module torchvision --exclude-module torchaudio `
    --exclude-module ultralytics --exclude-module ultralytics_thop `
    --exclude-module gradio --exclude-module gradio_client `
    --exclude-module matplotlib --exclude-module pandas --exclude-module scipy `
    --exclude-module PIL --exclude-module tkinter --exclude-module IPython `
    --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets `
    --exclude-module PySide6.QtWebEngineQuick --exclude-module PySide6.QtWebChannel `
    --exclude-module PySide6.QtPdf --exclude-module PySide6.QtPdfWidgets `
    --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick `
    --exclude-module PySide6.QtQuickWidgets --exclude-module PySide6.QtQuick3D `
    --exclude-module PySide6.QtMultimedia --exclude-module PySide6.QtMultimediaWidgets `
    --exclude-module PySide6.QtNetwork --exclude-module PySide6.QtNetworkAuth `
    --exclude-module PySide6.QtSql --exclude-module PySide6.QtTest `
    --exclude-module PySide6.QtHelp --exclude-module PySide6.QtDesigner `
    --exclude-module PySide6.QtUiTools --exclude-module PySide6.QtXml `
    --exclude-module PySide6.QtSvg --exclude-module PySide6.QtSvgWidgets `
    --exclude-module PySide6.QtOpenGL --exclude-module PySide6.QtOpenGLWidgets `
    --exclude-module PySide6.QtCharts --exclude-module PySide6.QtDataVisualization `
    --exclude-module PySide6.QtBluetooth --exclude-module PySide6.QtNfc `
    --exclude-module PySide6.QtPositioning --exclude-module PySide6.QtLocation `
    --exclude-module PySide6.QtSensors --exclude-module PySide6.QtSerialPort `
    --exclude-module PySide6.QtSerialBus --exclude-module PySide6.QtRemoteObjects `
    --exclude-module PySide6.QtScxml --exclude-module PySide6.QtTextToSpeech `
    --exclude-module PySide6.QtWebSockets --exclude-module PySide6.QtStateMachine `
    --exclude-module PySide6.Qt3DCore --exclude-module PySide6.Qt3DRender `
    --exclude-module PySide6.Qt3DInput --exclude-module PySide6.Qt3DAnimation `
    desktop.py

if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败" }

$exe = "dist\hat_detector\hat_detector.exe"
if (-not (Test-Path $exe)) { throw "没找到 $exe" }

# ---- 裁剪用不到的 Qt 组件 ----
# PyInstaller 的 PySide6 钩子会把整个 Qt 带进来，但本程序只用 QtCore/QtGui/QtWidgets。
# opengl32sw 是软件 OpenGL 回退（19.7MB），Qt6Quick/Qt6Qml/Qt6Pdf/Qt6OpenGL/虚拟键盘
# 这些都不被用到，删掉不影响启动（删完已实测 exe 正常）。
$internal = "dist\hat_detector\_internal"
$trim = @(
    "PySide6\opengl32sw.dll",
    "PySide6\Qt6Quick*.dll",
    "PySide6\Qt6Qml*.dll",
    "PySide6\Qt6Pdf*.dll",
    "PySide6\Qt6OpenGL*.dll",
    "PySide6\Qt6VirtualKeyboard*.dll",
    "PySide6\Qt6ShaderTools*.dll"
)
$freed = 0
foreach ($pattern in $trim) {
    Get-ChildItem -Path (Join-Path $internal $pattern) -ErrorAction SilentlyContinue | ForEach-Object {
        $freed += $_.Length
        Remove-Item $_.FullName -Force
    }
}
Write-Host ("裁剪无用 Qt 组件: {0:N1} MB" -f ($freed / 1MB))

$size = (Get-ChildItem "dist\hat_detector" -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "=== 打包完成 ==="
Write-Host "目录: $proj\dist\hat_detector"
Write-Host ("体积: {0:N1} MB" -f $size)
Write-Host "启动: 双击 $exe"
