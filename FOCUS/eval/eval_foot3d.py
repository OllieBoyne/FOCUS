"""Run evaluation on all of Foot3D dataset."""

import os
import re
from pathlib import Path
import pandas as pd

from FOCUS.eval.eval import evaluate_mesh

import argparse

parser = argparse.ArgumentParser(description="Evaluate a directory of meshes.")

parser.add_argument("--gt_dir", type=str, help="Path to ground truth meshes.")
parser.add_argument("--pred_dir", type=str, help="Path to predicted meshes.")
parser.add_argument(
    "--save_csv", action="store_true", help="Save evaluation results to a CSV file."
)
parser.add_argument(
    "--silent", action="store_true", help="Do not print results to console."
)
parser.add_argument("--use_caching", action="store_true", help="Use cached results.")
parser.add_argument("--fname", type=str, default="mesh", help="Name of the mesh files.")
parser.add_argument(
    "--is_find", action="store_true", help="Whether the mesh is from FIND."
)
parser.add_argument(
    "--save_evaluated_mesh",
    action="store_true",
    help="Whether to save the evaluated mesh.",
)


def eval_dir(
    gt_dir: str | Path,
    pred_dir: str | Path,
    save_csv=True,
    silent=True,
    use_caching: bool = False,
    fname: str = "mesh",
    is_find: bool = False,
    save_evaluated_mesh: bool = False,
) -> pd.DataFrame:
    """Evaluate a directory of meshes.

    Args:
        gt_dir: Path to ground truth meshes.
        pred_dir: Path to predicted meshes.
        save_csv: Whether to save evaluation results to a CSV file.
        silent: Whether to print results to console.
        use_caching: Whether to use cached results.
        fname: Name of the mesh files.
        is_find: Whether the mesh is from FIND.
        save_evaluated_mesh: Whether to save the evaluated mesh.
    :return: Evaluation results.
    """

    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    stats_dfs = {}

    csv_loc = pred_dir / "eval.csv"
    if use_caching and csv_loc.exists():
        return pd.read_csv(csv_loc, index_col=0)

    for f in sorted(os.listdir(pred_dir)):
        if re.fullmatch(r"\d{4}", f) is None:
            continue

        gt_src = gt_dir / f / "mesh.obj"
        pred_src = pred_dir / f / f"{fname}.obj"

        if not silent:
            print("-->", f)
        stats_dfs[f] = evaluate_mesh(
            gt_src,
            pred_src,
            silent=silent,
            use_caching=use_caching,
            is_find=is_find,
            save_evaluated_mesh=save_evaluated_mesh,
        )

    overall_df = pd.DataFrame(columns=list(stats_dfs.values())[0].columns)

    for key in sorted(stats_dfs.keys()):
        overall_df.loc[key] = stats_dfs[key].mean()

    if not silent:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)
        stats_df_print = overall_df.copy().round(1).astype(str)
        stats_df_print.loc["combined"] = overall_df.mean().round(1).astype(str)
        print("--> OVERALL")
        print(stats_df_print)

    if save_csv:
        overall_df.to_csv(csv_loc)

    return overall_df


if __name__ == "__main__":
    args = parser.parse_args()
    eval_dir(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        save_csv=args.save_csv,
        silent=args.silent,
        use_caching=args.use_caching,
        fname=args.fname,
        is_find=args.is_find,
        save_evaluated_mesh=args.save_evaluated_mesh,
    )
