from setuptools import setup, find_packages
import subprocess
import os

# Avoid issues compiling PyTorch3D on Mac.
os.environ['MAX_JOBS'] = '1'
skip_pytorch3d = os.environ.get('SKIP_PYTORCH3D', '0') == '1'

# Install PyTorch  & PyTorch3D first (necessary for PyTorch3D).
with open("requirements.txt") as f:
    for line in f.read().splitlines():
        if line.startswith(('torch==', 'torchvision==', 'torchaudio==')):
            subprocess.run(["pip", "install", line], check=True)

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

    if skip_pytorch3d:
        install_requires = [line for line in install_requires if not line.startswith('pytorch3d')]

setup(
    name='focus',
    version='0.1',
    packages=find_packages(),
    install_requires=install_requires,
)