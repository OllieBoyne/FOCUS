"""Compress all images in a folder using pngquant."""

import os
import sys
from pathlib import Path

quality='65-80'
folder = 'itw'
out_folder = f'{folder}_compressed'

# Get all sub files recursively
for file in Path(folder).rglob('*'):
    if file.is_file() and file.suffix in ['.png']:
        print(f'Compressing {file}')
        out_file = str(file).replace(folder, out_folder)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        os.system(f'pngquant --quality={quality} --force --output {out_file} {file}')