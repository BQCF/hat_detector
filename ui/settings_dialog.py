# ui/settings_dialog.py
# 设置面板：改模型、阈值、报警参数，确定后写回 configs/default.yaml。
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                               QVBoxLayout)

from core.config import AppConfig, save_config


class SettingsDialog(QDialog):
    """把 AppConfig 变成一张可编辑的表单。"""

    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("设置")
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)

        # ---------- 模型与推理 ----------
        g1 = QGroupBox("模型与推理")
        f1 = QFormLayout(g1)

        self.model_edit = QLineEdit(cfg.model_path or "")
        self.model_edit.setPlaceholderText("留空 = 自动使用最新的 best.pt")
        row = QHBoxLayout()
        row.addWidget(self.model_edit)
        btn = QPushButton("浏览…")
        btn.clicked.connect(self._pick_model)
        row.addWidget(btn)
        f1.addRow("模型路径", row)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "1"])
        self.device_combo.setCurrentText(cfg.device)
        f1.addRow("运行设备", self.device_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(cfg.conf)
        f1.addRow("置信度阈值", self.conf_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(cfg.imgsz)
        f1.addRow("推理分辨率", self.imgsz_spin)

        self.maxdet_spin = QSpinBox()
        self.maxdet_spin.setRange(10, 300)
        self.maxdet_spin.setValue(cfg.max_det)
        f1.addRow("最多检测框数", self.maxdet_spin)
        root.addWidget(g1)

        # ---------- 报警 ----------
        g2 = QGroupBox("报警")
        f2 = QFormLayout(g2)

        self.alarm_check = QCheckBox("启用报警")
        self.alarm_check.setChecked(cfg.alarm.enabled)
        f2.addRow(self.alarm_check)

        self.confirm_spin = QSpinBox()
        self.confirm_spin.setRange(1, 15)
        self.confirm_spin.setValue(cfg.alarm.confirm_frames)
        f2.addRow("连续几帧才确认", self.confirm_spin)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.2, 30.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setValue(cfg.alarm.cooldown_sec)
        f2.addRow("两次提醒最小间隔(秒)", self.cooldown_spin)

        self.sound_check = QCheckBox("报警时发声")
        self.sound_check.setChecked(cfg.alarm.sound)
        f2.addRow(self.sound_check)

        self.snap_check = QCheckBox("报警时抓拍存图")
        self.snap_check.setChecked(cfg.alarm.snapshot)
        f2.addRow(self.snap_check)

        self.dir_edit = QLineEdit(cfg.alarm.snapshot_dir)
        row2 = QHBoxLayout()
        row2.addWidget(self.dir_edit)
        btn2 = QPushButton("浏览…")
        btn2.clicked.connect(self._pick_dir)
        row2.addWidget(btn2)
        f2.addRow("抓拍/事件表目录", row2)
        root.addWidget(g2)

        # ---------- 按钮 ----------
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---------------- 交互 ----------------
    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", str(Path.cwd()), "PyTorch 权重 (*.pt);;所有文件 (*)")
        if path:
            self.model_edit.setText(path)

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择抓拍目录",
                                             self.dir_edit.text() or str(Path.cwd()))
        if d:
            self.dir_edit.setText(d)

    def _save_and_accept(self) -> None:
        """把表单内容写回配置并落盘。"""
        c = self.cfg
        c.model_path = self.model_edit.text().strip()
        c.device = self.device_combo.currentText()
        c.conf = self.conf_spin.value()
        c.imgsz = self.imgsz_spin.value()
        c.max_det = self.maxdet_spin.value()
        c.alarm.enabled = self.alarm_check.isChecked()
        c.alarm.confirm_frames = self.confirm_spin.value()
        c.alarm.cooldown_sec = self.cooldown_spin.value()
        c.alarm.sound = self.sound_check.isChecked()
        c.alarm.snapshot = self.snap_check.isChecked()
        c.alarm.snapshot_dir = self.dir_edit.text().strip() or "captures"
        save_config(c)
        self.accept()
