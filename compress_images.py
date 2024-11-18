"""Compress all files, resizing to a fixed size also."""

import os
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description='Compress all images in the current directory.')

parser.add_argument('--file', type=str, default=None, help='File to compress.')
parser.add_argument('--replace', action='store_true', help='Replace the original file.')


def main(file, replace=False):
    # Run recursively for directories
    if os.path.isdir(file):
        print('Compressing all images in the directory:', file)
        for f in os.listdir(file):
            main(os.path.join(file, f), replace)

    elif not os.path.isfile(file):
        raise FileNotFoundError('File not found.')

    commands = ['pngquant', '--quality=65', '-v']
    if replace:
        commands += [file, '--output', file, '-f']

    else:
        ext = os.path.splitext(file)[1]
        outfname = file.replace(ext, '_compressed' + ext)
        commands += [file, '--output', outfname]

    result = subprocess.run(commands, stdout=sys.stdout, stderr=sys.stderr, text=True)
    print(' '.join(commands))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.file, args.replace)