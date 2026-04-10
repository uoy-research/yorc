"""
File selection panel for YORC GUI.

Provides widgets for selecting input files with drag-and-drop support.
"""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FileSelector(QWidget):
    """Widget for selecting a single file."""

    file_selected = pyqtSignal(str)  # file path
    file_cleared = pyqtSignal()

    def __init__(self, label, file_filter="All Files (*)", parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self._setup_ui(label)

    def _setup_ui(self, label):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(100)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Drop file here or click Browse...")

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        self.clear_btn.setEnabled(False)

        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)
        layout.addWidget(self.clear_btn)

        # Enable drag and drop
        self.setAcceptDrops(True)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {self.label.text()}", "", self.file_filter
        )
        if path:
            self.set_path(path)

    def _on_clear(self):
        self.path_edit.clear()
        self.clear_btn.setEnabled(False)
        self.file_cleared.emit()

    def set_path(self, path):
        self.path_edit.setText(path)
        self.clear_btn.setEnabled(True)
        self.file_selected.emit(path)

    def get_path(self):
        return self.path_edit.text() or None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.set_path(path)


class MultiFileSelector(QWidget):
    """Widget for selecting multiple files."""

    files_changed = pyqtSignal(list)  # list of file paths

    def __init__(self, label, file_filter="All Files (*)", parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self._setup_ui(label)

    def _setup_ui(self, label):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QHBoxLayout()
        self.label = QLabel(label)
        header_layout.addWidget(self.label)
        header_layout.addStretch()

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._on_add)
        header_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        self.remove_btn.setEnabled(False)
        header_layout.addWidget(self.remove_btn)

        layout.addLayout(header_layout)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(80)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.file_list)

        # Enable drag and drop
        self.setAcceptDrops(True)

    def _on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select {self.label.text()}", "", self.file_filter
        )
        for path in paths:
            self._add_file(path)

    def _on_remove(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._emit_files_changed()

    def _on_selection_changed(self):
        self.remove_btn.setEnabled(len(self.file_list.selectedItems()) > 0)

    def _add_file(self, path):
        # Check for duplicates
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return

        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.file_list.addItem(item)
        self._emit_files_changed()

    def _emit_files_changed(self):
        paths = self.get_paths()
        self.files_changed.emit(paths)

    def get_paths(self):
        paths = []
        for i in range(self.file_list.count()):
            paths.append(self.file_list.item(i).data(Qt.ItemDataRole.UserRole))
        return paths

    def clear(self):
        self.file_list.clear()
        self._emit_files_changed()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path):
                self._add_file(path)


class FolderSelector(QWidget):
    """Widget for selecting a folder."""

    folder_selected = pyqtSignal(str)  # folder path
    folder_cleared = pyqtSignal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._setup_ui(label)

    def _setup_ui(self, label):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(100)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select folder...")

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        self.clear_btn.setEnabled(False)

        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)
        layout.addWidget(self.clear_btn)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, f"Select {self.label.text()}", "")
        if path:
            self.set_path(path)

    def _on_clear(self):
        self.path_edit.clear()
        self.clear_btn.setEnabled(False)
        self.folder_cleared.emit()

    def set_path(self, path):
        self.path_edit.setText(path)
        self.clear_btn.setEnabled(True)
        self.folder_selected.emit(path)

    def get_path(self):
        return self.path_edit.text() or None


class FilePanel(QWidget):
    """Panel containing all file selection widgets."""

    file_loaded = pyqtSignal(str, str)  # file_type, path
    file_cleared = pyqtSignal(str)  # file_type

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Input files group
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)

        self.outside_mesh = FileSelector("Outside Mesh:", "Mesh Files (*.ply *.stl);;All Files (*)")
        input_layout.addWidget(self.outside_mesh)

        self.inside_mesh = FileSelector("Inside Mesh:", "Mesh Files (*.ply *.stl);;All Files (*)")
        input_layout.addWidget(self.inside_mesh)

        self.mri_scalp = FileSelector("MRI Scalp:", "Mesh Files (*.stl *.ply);;All Files (*)")
        input_layout.addWidget(self.mri_scalp)

        self.meg_data = MultiFileSelector("MEG Data Files:", "FIF Files (*.fif);;All Files (*)")
        input_layout.addWidget(self.meg_data)

        layout.addWidget(input_group)

        # FreeSurfer group (optional)
        fs_group = QGroupBox("FreeSurfer (Optional - for automatic fiducials)")
        fs_layout = QVBoxLayout(fs_group)

        self.subjects_dir = FolderSelector("Subjects Dir:")
        fs_layout.addWidget(self.subjects_dir)

        self.subject_selector = FileSelector("Subject:", "All Files (*)")
        self.subject_selector.label.setText("Subject ID:")
        self.subject_selector.path_edit.setReadOnly(False)
        self.subject_selector.path_edit.setPlaceholderText("e.g., sub-01")
        self.subject_selector.browse_btn.setVisible(False)
        fs_layout.addWidget(self.subject_selector)

        layout.addWidget(fs_group)

    def _connect_signals(self):
        # Connect file selectors
        self.outside_mesh.file_selected.connect(lambda p: self.file_loaded.emit("outside_mesh", p))
        self.outside_mesh.file_cleared.connect(lambda: self.file_cleared.emit("outside_mesh"))

        self.inside_mesh.file_selected.connect(lambda p: self.file_loaded.emit("inside_mesh", p))
        self.inside_mesh.file_cleared.connect(lambda: self.file_cleared.emit("inside_mesh"))

        self.mri_scalp.file_selected.connect(lambda p: self.file_loaded.emit("mri_scalp", p))
        self.mri_scalp.file_cleared.connect(lambda: self.file_cleared.emit("mri_scalp"))

        self.meg_data.files_changed.connect(self._on_meg_files_changed)

        self.subjects_dir.folder_selected.connect(
            lambda p: self.file_loaded.emit("subjects_dir", p)
        )
        self.subjects_dir.folder_cleared.connect(lambda: self.file_cleared.emit("subjects_dir"))

    def _on_meg_files_changed(self, paths):
        self.file_loaded.emit("meg_data", ",".join(paths) if paths else "")

    def browse_all(self):
        """Open dialogs to browse for all files."""
        self.outside_mesh._on_browse()

    def get_all_paths(self):
        """Get all file paths as a dictionary."""
        return {
            "outside_mesh": self.outside_mesh.get_path(),
            "inside_mesh": self.inside_mesh.get_path(),
            "mri_scalp": self.mri_scalp.get_path(),
            "meg_data": self.meg_data.get_paths(),
            "subjects_dir": self.subjects_dir.get_path(),
            "subject": self.subject_selector.path_edit.text() or None,
        }
