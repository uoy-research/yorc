#!/usr/bin/env python3
"""Entry point for tri-panel registration GUI."""

import argparse
import sys

from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from yorc.gui.tripanel_registration_window import TriplePanelRegistrationWindow


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for optional GUI prefill paths."""
    parser = argparse.ArgumentParser(
        description="Launch tri-panel registration GUI with optional prefilled file paths."
    )
    parser.add_argument(
        "-im",
        "--inside-mesh",
        dest="inside_mesh",
        help="Inside LIDAR mesh/point cloud path (.ply/.stl/.obj/.pcd)",
    )
    parser.add_argument(
        "-om",
        "--outside-mesh",
        dest="outside_mesh",
        help="Outside LIDAR mesh/point cloud path (.ply/.stl/.obj/.pcd)",
    )
    parser.add_argument(
        "-s",
        "--mri-scalp",
        dest="mri_scalp",
        help="MRI scalp path (.fif/.ply/.stl/.obj)",
    )
    parser.add_argument(
        "-m",
        "--megdata",
        dest="meg_data",
        help="MEG FIF file path",
    )
    parser.add_argument(
        "--auto-load",
        action="store_true",
        help="Automatically run the 'Load Data' step on startup if inside/outside/MRI paths are set.",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "pyvista", "native-vtk"],
        default="auto",
        help=(
            "3D renderer backend. 'auto' prefers native-vtk on macOS for stability, "
            "otherwise pyvista."
        ),
    )
    return parser


def apply_cli_paths(window: TriplePanelRegistrationWindow, args: argparse.Namespace) -> None:
    """Populate GUI file fields from parsed CLI arguments."""
    if args.inside_mesh:
        window.inside_edit.setText(args.inside_mesh)
    if args.outside_mesh:
        window.outside_edit.setText(args.outside_mesh)
    if args.mri_scalp:
        window.mri_edit.setText(args.mri_scalp)
    if args.meg_data:
        window.meg_edit.setText(args.meg_data)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if sys.platform == "darwin":
        # Use software OpenGL on macOS: hardware OpenGL causes VTK to start a
        # CADisplayLink per widget (60fps x 3 panels = 180 renders/s), which
        # starves Qt's event loop so QTimer callbacks never fire. Software
        # rendering avoids this; renders are composited into Qt widgets via
        # pixmap correctly on macOS.
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

    # VTK + Qt recommendation: configure GL sharing and default surface format
    # before QApplication construction.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    if hasattr(QVTKRenderWindowInteractor, "defaultFormat"):
        QSurfaceFormat.setDefaultFormat(QVTKRenderWindowInteractor.defaultFormat())

    if sys.platform == "darwin":
        try:
            from AppKit import NSApp, NSApplication, NSApplicationActivationPolicyRegular

            NSApplication.sharedApplication()
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("YORC Tri-Panel Registration")
    app.setQuitOnLastWindowClosed(True)

    backend = args.renderer
    if backend == "auto":
        backend = "native-vtk" if sys.platform == "darwin" else "pyvista"
    viewer_backend = "native-vtk" if backend == "native-vtk" else "pyvista"

    window = TriplePanelRegistrationWindow(viewer_backend=viewer_backend)
    apply_cli_paths(window, args)

    has_all_required_inputs = bool(args.inside_mesh and args.outside_mesh and args.mri_scalp)
    should_auto_load = args.auto_load or has_all_required_inputs

    window.show()
    window.showNormal()
    print(
        f"YORC tri-panel GUI is running (renderer={backend}). "
        "Keep this terminal open while the window is open."
    )

    if sys.platform == "darwin":
        window.raise_()
        window.activateWindow()
        QTimer.singleShot(100, lambda: (window.raise_(), window.activateWindow()))
        QTimer.singleShot(350, lambda: (window.raise_(), window.activateWindow()))

    if should_auto_load:
        # Defer heavy mesh loading until after VTK renderers are initialized.
        # Software OpenGL context creation takes ~250 ms per panel × 3 panels,
        # so 1 second gives reliable headroom on all tested macOS hardware.
        print("Auto-load enabled: loading inside/outside/MRI inputs...")

        def _trigger_auto_load() -> None:
            print("Auto-load callback fired")
            try:
                window.load_data()
            except Exception as exc:
                print(f"Auto-load callback error: {exc}")

        QTimer.singleShot(1000, _trigger_auto_load)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
