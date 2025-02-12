"""Run COLMAP using our TOC matches."""
import os

from FOCUS.matching import correspondence, match
from FOCUS.data.dataset import load_views
from pathlib import Path
from FOCUS.calibration.colmap_database import COLMAPDatabase
import numpy as np
from collections import defaultdict

_IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg')

def toc_matches_to_database(predictions_dir: Path, output_dir: Path, num_correspondences=2500) -> list[str]:
    """Using TOC matches, form COLMAP database for BA.

    Returns list of image ids"""

    views = load_views(predictions_dir)

    correspondences = match.find_matches(views, num_correspondences=num_correspondences,
                                         max_dist=0.002,
                                         subpixel_scaling=8)

    # Create a new database
    output_dir.mkdir(exist_ok=True, parents=True)
    db_path = output_dir / 'database.db'
    os.remove(db_path) if db_path.exists() else None

    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # Add cameras
    # Specify the camera parameters
    image_height, image_width = views[0].rgb.shape[:2]
    model1, width1, height1, params1 = (
        0, # SIMPLE_PINHOLE
        image_width,
        image_height,
        np.array((768, image_width / 2, image_height / 2)), # f, cx, cy
    )
    camera_id1 = db.add_camera(model1, width1, height1, params1)

    img_ids = []
    for v in views:
        img_ids.append(v.key)
        db.add_image(v.key + '/rgb.png', camera_id1, image_id=v.idx)

    # per view, a mapping of correspondence index -> keypoint index within the image
    correspondence_to_image_idx = defaultdict(dict)

    keypoints_per_view = defaultdict(list)

    for cidx, correspondence in enumerate(correspondences):
        for i, vidx in enumerate(correspondence.idxs):
            keypoints_per_view[vidx].append(correspondence.points_2d[i])

            view_kp_idx = len(correspondence_to_image_idx[vidx])
            correspondence_to_image_idx[vidx][cidx] = view_kp_idx

    for v in views:
        kpts = keypoints_per_view[v.idx]
        kpts = np.array(kpts)
        db.add_keypoints(v.idx, kpts)

    # get all pairs of views
    from itertools import combinations
    for v1, v2 in combinations(views, 2):
        match_1 = []
        match_2 = []

        for cidx, correspondence in enumerate(correspondences):
            if cidx in correspondence_to_image_idx[v1.idx] and cidx in correspondence_to_image_idx[v2.idx]:
                match_1.append(correspondence_to_image_idx[v1.idx][cidx])
                match_2.append(correspondence_to_image_idx[v2.idx][cidx])

        match_array = np.array([match_1, match_2]).T
        db.add_matches(v1.idx, v2.idx, match_array)
        db.add_two_view_geometry(v1.idx, v2.idx, match_array)

    db.commit()
    db.close()

    return img_ids
