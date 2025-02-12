<p align="center">
  <a href="http://ollieboyne.github.io/FOCUS">
        <img width=70% src="https://www.ollieboyne.com/FOCUS/images/logos/focus_v1.png">
  </a>
</p>

This repository contains the code for foot reconstruction using dense correspondences, as shown in our paper:

> **FOCUS: Multi-View Foot Reconstruction from Synthetically Trained Dense Correspondences**  \
> 3DV 2025 \
> [Oliver Boyne](https://ollieboyne.github.io) and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2502.06367) [[project page]](https://ollieboyne.github.io/FOCUS/)

## Set-up

1) `git clone --recurse-submodules https://github.com/OllieBoyne/FOCUS.git`
2) Install dependencies: `pip install -r requirements.txt`
3) Install FOCUS: `pip install --editable .`
4) Download any required data (see Downloads section).
5) If using uncalibrated images, install [COLMAP](https://colmap.github.io/install.html).

## Usage

FOCUS can be run using the `FOCUS/run_focus.py` script:

- Use the `--method` flag to choose between `sfm` (FOCUS-SfM) and `o` (FOCUS-O).
- For a directory of images, use the `--source_folder` flag.
- For a video, use the `--video_path` flag.

See the `run_focus.py` script for more options.


## Downloads

| Item                                                                                               | Description                                    | Download to      |
|----------------------------------------------------------------------------------------------------|------------------------------------------------|------------------|
| [TOC model](https://drive.google.com/file/d/1aU1Bf_pE7WjtWAX85ru9htQGygmTDIUV/view?usp=share_link) | A pre-trained correspondence predictor model.  | `data/toc_model` |
| [Foot 3D](https://github.com/OllieBoyne/Foot3D)                                                    | Multi-view foot dataset for evaluating method. | `data/Foot3D`     |
| [3D Fits](https://drive.google.com/file/d/1B0V5sRUBkj9kjv-q45jjYdSDiFEe3j-o/view?usp=share_link)   | 3D reconstructions for evaluation, as used in the paper. | `data/3d_fits`     |

### Citation

If you use our work, please cite:

```
@inproceedings{boyne2025focus,
            title={FOCUS: Multi-View Foot Reconstruction from Synthetically Trained Dense Correspondences},
            author={Boyne, Oliver and Cipolla, Roberto},
            booktitle={2025 International Conference on 3D Vision (3DV)},
            year={2025}
}
```