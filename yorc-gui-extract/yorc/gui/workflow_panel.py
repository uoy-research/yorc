"""
Workflow panel for YORC GUI.

Provides step-by-step workflow controls and status indicators.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StepIndicator(QWidget):
    """Widget showing status of a single workflow step."""

    run_clicked = pyqtSignal(int)  # step index

    # Status constants
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self, step_index, title, description="", parent=None):
        super().__init__(parent)
        self.step_index = step_index
        self.status = self.PENDING
        self._setup_ui(title, description)

    def _setup_ui(self, title, description):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Status indicator
        self.status_label = QLabel("○")
        self.status_label.setFixedWidth(20)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Step info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self.title_label = QLabel(title)
        font = QFont()
        font.setBold(True)
        self.title_label.setFont(font)
        info_layout.addWidget(self.title_label)

        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet("color: gray; font-size: 11px;")
            info_layout.addWidget(self.desc_label)
        else:
            self.desc_label = None

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("font-size: 10px;")
        self.message_label.setVisible(False)
        info_layout.addWidget(self.message_label)

        layout.addLayout(info_layout, 1)

        # Run button
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedWidth(60)
        self.run_btn.clicked.connect(lambda: self.run_clicked.emit(self.step_index))
        layout.addWidget(self.run_btn)

        self._update_style()

    def set_status(self, status, message=""):
        """Update step status."""
        self.status = status
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        self._update_style()

    def _update_style(self):
        """Update visual style based on status."""
        if self.status == self.PENDING:
            self.status_label.setText("○")
            self.status_label.setStyleSheet("color: gray;")
            self.run_btn.setEnabled(True)
        elif self.status == self.RUNNING:
            self.status_label.setText("◐")
            self.status_label.setStyleSheet("color: blue;")
            self.run_btn.setEnabled(False)
        elif self.status == self.COMPLETED:
            self.status_label.setText("●")
            self.status_label.setStyleSheet("color: green;")
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Redo")
        elif self.status == self.FAILED:
            self.status_label.setText("✗")
            self.status_label.setStyleSheet("color: red;")
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Retry")

    def set_enabled(self, enabled):
        """Enable or disable the step."""
        self.run_btn.setEnabled(enabled and self.status != self.RUNNING)


class WorkflowPanel(QWidget):
    """Panel containing workflow steps and controls."""

    run_automatic = pyqtSignal()
    run_step = pyqtSignal(int)  # step index
    cancel_requested = pyqtSignal()

    STEPS = [
        ("Step 1: Helmet Detection", "Detect colored roundels on helmet"),
        ("Step 2: Anatomical Points", "Define head coordinate system"),
        ("Step 3: Head-to-Head", "Register helmet scan to head scan"),
        ("Step 4: Head-to-MRI", "Register MRI scalp to head scan"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Workflow group
        workflow_group = QGroupBox("Registration Workflow")
        workflow_layout = QVBoxLayout(workflow_group)

        # Automatic run button
        auto_layout = QHBoxLayout()

        self.auto_btn = QPushButton("Run Automatic")
        self.auto_btn.setMinimumHeight(40)
        font = QFont()
        font.setBold(True)
        self.auto_btn.setFont(font)
        self.auto_btn.clicked.connect(self._on_auto_clicked)
        auto_layout.addWidget(self.auto_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        auto_layout.addWidget(self.cancel_btn)

        workflow_layout.addLayout(auto_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        workflow_layout.addWidget(line)

        # Manual steps label
        manual_label = QLabel("Or run steps individually:")
        manual_label.setStyleSheet("color: gray;")
        workflow_layout.addWidget(manual_label)

        # Step indicators
        self.step_indicators = []
        for i, (title, desc) in enumerate(self.STEPS):
            indicator = StepIndicator(i, title, desc)
            indicator.run_clicked.connect(self._on_step_clicked)
            self.step_indicators.append(indicator)
            workflow_layout.addWidget(indicator)

        layout.addWidget(workflow_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.current_step_label = QLabel("Ready")
        progress_layout.addWidget(self.current_step_label)

        self.step_progress = QProgressBar()
        self.step_progress.setVisible(False)
        progress_layout.addWidget(self.step_progress)

        layout.addWidget(progress_group)

    def _on_auto_clicked(self):
        """Handle automatic run button click."""
        self.run_automatic.emit()

    def _on_step_clicked(self, step_index):
        """Handle individual step run button click."""
        self.run_step.emit(step_index)

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()

    def set_running(self, running):
        """Set running state of the workflow."""
        self.is_running = running
        self.auto_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

        for indicator in self.step_indicators:
            indicator.set_enabled(not running)

        if running:
            self.current_step_label.setText("Running...")
            self.step_progress.setVisible(True)
        else:
            self.step_progress.setVisible(False)

    def set_step_status(self, step_index, success, message=""):
        """Update status of a specific step."""
        if 0 <= step_index < len(self.step_indicators):
            status = StepIndicator.COMPLETED if success else StepIndicator.FAILED
            self.step_indicators[step_index].set_status(status, message)

    def set_step_running(self, step_index):
        """Mark a step as currently running."""
        if 0 <= step_index < len(self.step_indicators):
            self.step_indicators[step_index].set_status(StepIndicator.RUNNING)
            self.current_step_label.setText(self.STEPS[step_index][0])

    def set_progress(self, percent, message=""):
        """Update progress bar."""
        self.step_progress.setValue(percent)
        if message:
            self.current_step_label.setText(message)

    def set_enabled(self, enabled):
        """Enable or disable the entire panel."""
        self.auto_btn.setEnabled(enabled and not self.is_running)
        for indicator in self.step_indicators:
            indicator.set_enabled(enabled and not self.is_running)

    def reset(self):
        """Reset all steps to pending state."""
        for indicator in self.step_indicators:
            indicator.set_status(StepIndicator.PENDING)
        self.current_step_label.setText("Ready")
        self.step_progress.setValue(0)
        self.step_progress.setVisible(False)
