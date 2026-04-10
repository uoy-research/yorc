#!/usr/bin/env python3
"""
YORC GUI Entry Point

Launch the YORC graphical user interface for OPM-MEG coregistration.

Usage:
    python yorc_gui.py

Or with command-line arguments:
    python yorc_gui.py --outside-mesh head.ply --inside-mesh helmet.ply
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="YORC GUI - York OPM Registration Code",
        epilog="""
Examples:
  # Load meshes on startup for testing
  python yorc_gui.py --outside-mesh head.ply --inside-mesh helmet.ply

  # Load all files
  python yorc_gui.py -om head.ply -im helmet.ply -s mri_scalp.stl -m meg_data.fif
        """
    )
    parser.add_argument(
        "-om", "--outside-mesh",
        help="LIDAR scan of head outside the MEG Helmet (.ply/.stl) - loads on startup"
    )
    parser.add_argument(
        "-im", "--inside-mesh",
        help="LIDAR scan of head inside the MEG Helmet (.ply/.stl) - loads on startup"
    )
    parser.add_argument(
        "-s", "--mri-scalp",
        help="MRI scalp surface from FreeSurfer (.stl) - loads on startup"
    )
    parser.add_argument(
        "-m", "--megdata",
        nargs='+',
        help="MEG data file(s) (.fif) - loads on startup"
    )
    parser.add_argument(
        "--subjects-dir",
        help="FreeSurfer subjects directory"
    )
    parser.add_argument(
        "--subject",
        help="FreeSurfer subject ID"
    )

    args = parser.parse_args()

    # macOS: Make the app a proper foreground application
    if sys.platform == 'darwin':
        try:
            from AppKit import NSApp, NSApplication, NSApplicationActivationPolicyRegular
            NSApplication.sharedApplication()
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            # pyobjc not installed, try alternative
            pass

    # Import Qt and GUI components
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        print("Error: PyQt6 is required for the GUI.")
        print(f"Python executable: {sys.executable}")
        print(f"Import error: {exc}")
        print("Install with: uv pip install --python .venv/bin/python PyQt6")
        sys.exit(1)

    try:
        import pyvista
        import pyvistaqt
    except ImportError as exc:
        print("Error: PyVista and PyVistaQt are required for 3D visualization.")
        print(f"Python executable: {sys.executable}")
        print(f"Import error: {exc}")
        print("Install with: uv pip install --python .venv/bin/python pyvista pyvistaqt vtk")
        print("Then launch with: source .venv/bin/activate && uv run yorc-gui")
        sys.exit(1)

    # Add parent directory to path for imports
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from yorc.gui import MainWindow

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("YORC")
    app.setOrganizationName("York Neuroimaging Centre")

    # macOS: Ensure app appears in dock and can receive focus
    if sys.platform == 'darwin':
        # This helps macOS recognize it as a proper GUI app
        app.setQuitOnLastWindowClosed(True)

    # Create main window
    window = MainWindow()

    # Show window first
    window.show()

    # Pre-load files if provided via command line (after window is shown)
    if args.outside_mesh:
        window.file_panel.outside_mesh.set_path(args.outside_mesh)
    if args.inside_mesh:
        window.file_panel.inside_mesh.set_path(args.inside_mesh)
    if args.mri_scalp:
        window.file_panel.mri_scalp.set_path(args.mri_scalp)
    if args.megdata:
        for path in args.megdata:
            window.file_panel.meg_data._add_file(path)
    if args.subjects_dir:
        window.file_panel.subjects_dir.set_path(args.subjects_dir)
    if args.subject:
        window.file_panel.subject_selector.path_edit.setText(args.subject)

    # macOS: Force window to front and make it focusable
    if sys.platform == 'darwin':
        window.raise_()
        window.activateWindow()
        # Additional macOS hack to bring window to front
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: (window.raise_(), window.activateWindow()))

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
