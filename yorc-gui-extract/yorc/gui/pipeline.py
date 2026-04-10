"""
Registration pipeline for YORC GUI.

Orchestrates the full registration workflow with progress callbacks.
"""

import numpy as np


class RegistrationPipeline:
    """
    Orchestrates the YORC registration pipeline.

    Manages the 4-step registration process with support for:
    - Fully automatic execution
    - Step-by-step manual execution
    - Progress callbacks for GUI updates
    - Intermediate result storage
    """

    def __init__(
        self,
        outside_mesh,
        inside_mesh,
        mri_scalp,
        meg_data,
        subjects_dir=None,
        subject=None,
        automatic=True,
        step=None,
        outside_cloud=None,
        inside_cloud=None,
        mri_cloud=None,
    ):
        """
        Initialize the registration pipeline.

        Args:
            outside_mesh: Path to outside head mesh (.ply)
            inside_mesh: Path to inside helmet mesh (.ply)
            mri_scalp: Path to MRI scalp surface (.stl)
            meg_data: List of paths to MEG data files (.fif)
            subjects_dir: Optional FreeSurfer subjects directory
            subject: Optional FreeSurfer subject ID
            automatic: Whether to run in fully automatic mode
            step: Specific step to run (0-3), or None for all steps
            outside_cloud: Pre-loaded outside point cloud (optional)
            inside_cloud: Pre-loaded inside point cloud (optional)
            mri_cloud: Pre-loaded MRI point cloud (optional)
        """
        self.outside_mesh = outside_mesh
        self.inside_mesh = inside_mesh
        self.mri_scalp = mri_scalp
        self.meg_data = meg_data if isinstance(meg_data, list) else [meg_data]
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.automatic = automatic
        self.target_step = step

        # Results storage
        self.results = {
            "X1": None,  # Helmet-to-device transform
            "X2": None,  # Head-to-head transform
            "X21": None,  # Combined device-to-head transform
            "X3": None,  # MRI-to-head transform
            "helmet_landmarks": None,
            "anatomical_points": None,
            "standard_trans": None,
            "standard_head": None,
            "mri_cloud": None,
            "errors": {},
        }

        # Loaded data (use pre-loaded if provided)
        self.inside_cloud = inside_cloud
        self.outside_cloud = outside_cloud
        self.mri_cloud = mri_cloud

    def run(self, progress_callback=None, step_callback=None, cancel_check=None):
        """
        Run the registration pipeline.

        Args:
            progress_callback: Function(message, percent) for progress updates
            step_callback: Function(step_index, success, message) for step completion
            cancel_check: Function() that returns True if cancelled
        """
        from ..core.fiducial_estimation import (
            estimate_fiducials_from_mri,
            fiducials_to_anatomical_array,
        )
        from ..core.io_utils import load_mesh
        from ..core.registration import (
            check_headpoints,
            head_to_head,
            head_to_helmet,
            head_to_mri,
            head_to_standard,
            write_output,
        )

        def report_progress(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        def report_step(idx, success, msg):
            if step_callback:
                step_callback(idx, success, msg)

        def is_cancelled():
            return cancel_check() if cancel_check else False

        try:
            # Load meshes (only if not pre-loaded)
            if self.inside_cloud is None:
                report_progress("Loading inside mesh...", 5)
                self.inside_cloud, _ = load_mesh(self.inside_mesh)
                if is_cancelled():
                    return

            if self.outside_cloud is None:
                report_progress("Loading outside mesh...", 10)
                self.outside_cloud, _ = load_mesh(self.outside_mesh)
                if is_cancelled():
                    return

            if self.mri_cloud is None:
                report_progress("Loading MRI mesh...", 15)
                self.mri_cloud, _ = load_mesh(self.mri_scalp)
                if is_cancelled():
                    return
            else:
                report_progress("Using pre-loaded meshes...", 15)

            # Determine which steps to run
            if self.target_step is not None:
                steps_to_run = [self.target_step]
            else:
                steps_to_run = [0, 1, 2, 3]

            # Step 0: Helmet Detection
            if 0 in steps_to_run:
                report_progress("Step 1: Detecting helmet landmarks...", 10)

                def helmet_progress(msg, pct):
                    report_progress(f"Step 1: {msg}", 10 + int(pct * 0.2))

                X1, landmarks, errors = head_to_helmet(
                    self.inside_cloud, progress_callback=helmet_progress
                )

                if X1 is None:
                    report_step(0, False, "Landmark detection failed")
                    if self.automatic:
                        raise RuntimeError(
                            "Automatic landmark detection failed. Please use manual mode."
                        )
                    return
                else:
                    self.results["X1"] = X1
                    self.results["helmet_landmarks"] = landmarks
                    self.results["errors"]["helmet"] = errors
                    error_str = f"RMSE: {np.mean(errors):.2f}mm"
                    report_step(0, True, error_str)

                if is_cancelled():
                    return

            # Step 1: Anatomical Points
            if 1 in steps_to_run:
                report_progress("Step 2: Defining anatomical coordinate system...", 30)

                anatomical_points = None

                # Try automatic fiducial estimation if FreeSurfer data available
                if self.automatic and self.subjects_dir and self.subject:
                    report_progress("Step 2: Estimating fiducials from FreeSurfer...", 32)
                    fiducials = estimate_fiducials_from_mri(self.subjects_dir, self.subject)

                    if fiducials:
                        anatomical_points = fiducials_to_anatomical_array(fiducials)
                        report_progress("Step 2: Fiducials estimated automatically", 35)

                if anatomical_points is None and not self.automatic:
                    # Manual mode - anatomical_points should be provided externally
                    report_step(1, False, "Manual point selection required")
                    return

                if anatomical_points is None:
                    # Automatic mode without FreeSurfer - this is a limitation
                    report_step(1, False, "No FreeSurfer data - manual selection needed")
                    raise RuntimeError(
                        "Automatic mode requires FreeSurfer subject data for fiducial estimation. "
                        "Please provide subjects_dir and subject, or use manual mode."
                    )

                standard_trans, standard_head = head_to_standard(
                    self.outside_cloud, anatomical_points
                )

                self.results["anatomical_points"] = anatomical_points
                self.results["standard_trans"] = standard_trans
                self.results["standard_head"] = standard_head
                report_step(1, True, "Coordinate system defined")

                if is_cancelled():
                    return

            # Step 2: Head-to-Head Registration
            if 2 in steps_to_run:
                report_progress("Step 3: Registering helmet to head scan...", 50)

                if self.results["standard_head"] is None:
                    report_step(2, False, "Step 2 must be completed first")
                    return

                def h2h_progress(msg, pct):
                    report_progress(f"Step 3: {msg}", 50 + int(pct * 0.2))

                X2, fitness = head_to_head(
                    self.results["standard_head"], self.inside_cloud, progress_callback=h2h_progress
                )

                self.results["X2"] = X2

                # Compute combined transform
                if self.results["X1"] is not None:
                    self.results["X21"] = np.dot(X2, self.results["X1"])

                fitness_str = f"Fitness: {fitness:.4f}"
                report_step(2, True, fitness_str)

                if is_cancelled():
                    return

            # Step 3: Head-to-MRI Registration
            if 3 in steps_to_run:
                report_progress("Step 4: Registering MRI to head scan...", 70)

                if self.results["standard_head"] is None:
                    report_step(3, False, "Step 2 must be completed first")
                    return

                def mri_progress(msg, pct):
                    report_progress(f"Step 4: {msg}", 70 + int(pct * 0.2))

                transformed_mri, X3, fitness = head_to_mri(
                    self.results["standard_head"], self.mri_cloud, progress_callback=mri_progress
                )

                self.results["X3"] = X3
                self.results["mri_cloud"] = transformed_mri

                fitness_str = f"Fitness: {fitness:.4f}"
                report_step(3, True, fitness_str)

                if is_cancelled():
                    return

            # Write outputs
            if self.results["X21"] is not None and self.results["X3"] is not None:
                report_progress("Writing outputs...", 90)

                for i, meg_path in enumerate(self.meg_data):
                    report_progress(
                        f"Processing MEG file {i + 1}/{len(self.meg_data)}...",
                        90 + int(10 * i / len(self.meg_data)),
                    )

                    # Validate registration
                    sensor_cloud, median_dist, distances = check_headpoints(
                        self.results["mri_cloud"], meg_path, self.results["X21"]
                    )
                    self.results["errors"][f"sensors_{i}"] = {
                        "median_distance": median_dist,
                        "distances": distances,
                    }

                    # Write output files
                    output_path = write_output(meg_path, self.results["X21"], self.results["X3"])
                    report_progress(
                        f"Written: {output_path}", 95 + int(5 * (i + 1) / len(self.meg_data))
                    )

                report_progress("Registration complete!", 100)

        except Exception as e:
            report_progress(f"Error: {str(e)}", 0)
            raise

    def get_step_result(self, step_index):
        """
        Get the result of a specific step for visualization.

        Args:
            step_index: Step index (0-3)

        Returns:
            Dict with visualization data, or None
        """
        if step_index == 0:
            return {
                "landmarks": self.results.get("helmet_landmarks"),
                "errors": self.results.get("errors", {}).get("helmet"),
            }
        elif step_index == 1:
            return {
                "anatomical_points": self.results.get("anatomical_points"),
                "transformed_cloud": self.results.get("standard_head"),
            }
        elif step_index == 2:
            return {
                "X21": self.results.get("X21"),
            }
        elif step_index == 3:
            return {
                "mri_cloud": self.results.get("mri_cloud"),
                "X3": self.results.get("X3"),
            }
        return None

    def set_anatomical_points(self, points):
        """
        Set manually picked anatomical points.

        Args:
            points: Array of 3 points [R_preauricular, L_preauricular, nasion]
        """
        self.results["anatomical_points"] = np.array(points)

    def get_transforms(self):
        """
        Get all computed transforms.

        Returns:
            Dict with 'X1', 'X2', 'X21', 'X3' keys
        """
        return {
            "X1": self.results.get("X1"),
            "X2": self.results.get("X2"),
            "X21": self.results.get("X21"),
            "X3": self.results.get("X3"),
        }
