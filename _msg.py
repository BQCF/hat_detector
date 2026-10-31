# 临时：单独渲染一个 QMessageBox，验证暗色样式是否生效
import sys

sys.path.insert(0, r"D:\projects\hat_detector")

from PySide6.QtWidgets import QApplication, QMessageBox

from ui import theme

app = QApplication(sys.argv)
app.setStyleSheet(theme.stylesheet())
QMessageBox.warning(None, "启动失败", "FileNotFoundError\n\n找不到模型权重，请先训练。")
