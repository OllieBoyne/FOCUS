"""Run evaluation on all 3D reconstructions compared in the FOCUS paper."""

import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm

script = 'FOCUS/eval/eval_foot3d.py'
data_source = 'data/paper_fits'
gt_source = 'data/Foot3D'


folders = [f for f in os.listdir(data_source) if os.path.isdir(os.path.join(data_source, f))]

assert len(folders) > 0, "No folders found in data source."
assert len(os.listdir(data_source)) > 0, "Foot3D not downloaded."

overall_df = pd.DataFrame(columns=folders)

with tqdm(folders) as tqdm_it:
    for folder in tqdm_it:
        tqdm_it.set_description(f"Evaluating {folder}")

        # FIND-based models require slightly different evaluation for fairness.
        is_find = folder in ['FOUND', 'FOCUS_O']

        subprocess.run(['python', script, '--gt_dir', gt_source,
                            '--pred_dir', os.path.join(data_source, folder),
                            '--silent',
                            '--save_csv',
                            '--save_evaluated_mesh'] \
                             + (['--is_find'] if is_find else []),
                       stdout=sys.stdout)

        out_data = pd.read_csv(os.path.join(data_source, folder, 'eval.csv'))
        overall_df[folder] = out_data.mean().round(1).astype(str)

overall_df = overall_df.iloc[1:,].reindex().transpose() # Remove 'name' row.

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
print(overall_df)