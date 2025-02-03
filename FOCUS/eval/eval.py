"""Functions for evaluating meshes."""

import os

import numpy
import trimesh
import numpy as np
from pathlib import Path
import dataclasses
import pandas as pd


def cutoff_slice_FIND(mesh: trimesh.Trimesh, height: float = 0.1):
    """Cutoff at FIND height."""
    return mesh.slice_plane([0, 0, height], [0, 0, -1], cap=False)


def sample(mesh: trimesh.Trimesh, num_points: int) -> (np.ndarray, np.ndarray):
    """Sample points from a mesh."""
    sample_points, sample_faces = trimesh.sample.sample_surface(mesh, num_points)

    return sample_points, mesh.face_normals[sample_faces]


def angle_between_normals(normals_1, normals_2) -> np.ndarray:
    """Return angle between normals, in degrees."""
    return np.rad2deg(np.arccos((normals_1 * normals_2).sum(axis=-1)))


@dataclasses.dataclass
class ChamferStat:
    """Chamfer statistics."""

    pred2real: list[numpy.ndarray]
    real2pred: list[numpy.ndarray]

    def __post_init__(self):
        if isinstance(self.pred2real, numpy.ndarray):
            self.pred2real = [self.pred2real]

        if isinstance(self.real2pred, numpy.ndarray):
            self.real2pred = [self.real2pred]

    @property
    def mean(self):
        vals = (
            np.mean([p.mean() for p in self.pred2real]),
            np.mean([r.mean() for r in self.real2pred]),
        )
        return np.mean(vals)

    @property
    def median(self):
        vals = (
            np.mean([np.median(p) for p in self.pred2real]),
            np.mean([np.median(r) for r in self.real2pred]),
        )
        return np.mean(vals)

    @property
    def rmse(self):
        vals = (
            np.mean([np.sqrt(np.mean(p**2)) for p in self.pred2real]),
            np.mean([np.sqrt(np.mean(r**2)) for r in self.real2pred]),
        )
        return np.mean(vals)

    def as_string(self, multiplier=1.0):
        return f"Mean: {self.mean * multiplier:.1f}, Median: {self.median * multiplier:.1f}, RMS: {self.rmse * multiplier:.1f}"


def evaluate_mesh(
    gt_obj_path: str | Path,
    pred_obj_path: str | Path,
    save_csv=True,
    silent=True,
    use_caching: bool = False,
    is_find=False,
    save_evaluated_mesh: bool = False,
) -> pd.DataFrame:
    """Evaluate a single mesh.

    Args:
        gt_obj_path: Path to ground truth mesh.
        pred_obj_path: Path to predicted mesh.
        save_csv: Whether to save evaluation results to a CSV file.
        silent: Whether to print results to console.
        use_caching: Whether to use cached results.
        is_find: Whether the mesh is from FIND.
        save_evaluated_mesh: Whether to save the evaluated mesh.
    :return: Evaluation results.
    """

    csv_loc = pred_obj_path.parent / "eval.csv"

    if use_caching and csv_loc.exists():
        return pd.read_csv(csv_loc, index_col=0)

    with open(gt_obj_path) as infile:
        d = trimesh.exchange.obj.load_obj(infile, process=False)
        gt_mesh_trimesh = trimesh.Trimesh(**d)
        gt_mesh_trimesh = cutoff_slice_FIND(gt_mesh_trimesh)

    with open(pred_obj_path) as infile:
        d = trimesh.exchange.obj.load_obj(infile, process=False)
        pred_mesh_trimesh = trimesh.Trimesh(**d)

        if is_find:
            FIND_cutoff_surface = np.load(
                os.path.join("data/find", "templ_masked_faces.npy")
            )
            FIND_sole_surface = np.load(
                os.path.join("data/find", "templ_sole_faces.npy")
            )
            all_masked_faces = np.concatenate([FIND_cutoff_surface, FIND_sole_surface])

            pred_mesh_trimesh.update_faces(
                ~np.isin(np.arange(pred_mesh_trimesh.faces.shape[0]), all_masked_faces)
            )

            # Also cut off below floor
            pred_mesh_trimesh = pred_mesh_trimesh.slice_plane(
                [0, 0, 0.0], [0, 0, 1], cap=False
            )

        pred_mesh_trimesh = cutoff_slice_FIND(pred_mesh_trimesh)

    np.random.seed(0)
    num_samples = 10_000
    gt_points, gt_normals = sample(gt_mesh_trimesh, num_samples)
    pred_points, pred_normals = sample(pred_mesh_trimesh, num_samples)

    _, cham_x, face_x = trimesh.proximity.closest_point(gt_mesh_trimesh, pred_points)
    _, cham_y, face_y = trimesh.proximity.closest_point(pred_mesh_trimesh, gt_points)

    cham_norm_pred2real = angle_between_normals(
        gt_mesh_trimesh.face_normals[face_x], pred_normals
    )
    cham_norm_real2pred = angle_between_normals(
        pred_mesh_trimesh.face_normals[face_y], gt_normals
    )

    cham_pred2real = cham_x * 1e3  # in mm.
    cham_real2pred = cham_y * 1e3  # in mm.

    if save_evaluated_mesh:
        # Compute per-vertex chamf & normal errors to save along with mesh.
        _, eval_chamf, eval_target_faces = trimesh.proximity.closest_point(
            gt_mesh_trimesh, pred_mesh_trimesh.vertices
        )
        eval_normals = angle_between_normals(
            gt_mesh_trimesh.face_normals[eval_target_faces],
            pred_mesh_trimesh.vertex_normals,
        )

        # Export to obj.
        eval_mesh = pred_mesh_trimesh.copy()
        eval_mesh.vertex_attributes["chamfer_error"] = eval_chamf
        eval_mesh.vertex_attributes["normal_error"] = eval_normals
        eval_mesh.export(pred_obj_path.parent / f"{pred_obj_path.stem}_eval.glb")

    stats_df = pd.DataFrame(
        index=["pred2real", "real2pred"],
        data={
            "chamfer_mean": [cham_pred2real.mean(), cham_real2pred.mean()],
            "chamfer_median": [np.median(cham_pred2real), np.median(cham_real2pred)],
            "chamfer_rmse": [
                np.sqrt(np.mean(cham_pred2real**2)),
                np.sqrt(np.mean(cham_real2pred**2)),
            ],
            "normal_mean": [cham_norm_pred2real.mean(), cham_norm_real2pred.mean()],
            "normal_median": [
                np.median(cham_norm_pred2real),
                np.median(cham_norm_real2pred),
            ],
            "normal_rmse": [
                np.sqrt(np.mean(cham_norm_pred2real**2)),
                np.sqrt(np.mean(cham_norm_real2pred**2)),
            ],
        },
    )

    if save_csv:
        stats_df.to_csv(csv_loc)

    if not silent:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)
        stats_df_print = stats_df.copy().round(1).astype(str)
        stats_df_print.loc["combined"] = stats_df.mean().round(1).astype(str)
        print(stats_df_print.head(3))

    return stats_df
