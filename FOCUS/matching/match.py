"""Find matches across Views."""

from FOCUS.matching import view
from FOCUS.matching import correspondence
import torch

def find_matches(views: [view.View], num_correspondences: int,
                 max_dist: float = 0.002,
                 subpixel_scaling: int = 8
                 ) -> [correspondence.Correspondence]:
    """Find matches across Views."""

    correspondences: list[correspondence.Correspondence] = []

    torch.manual_seed(10) # TODO: add seed to hyperparameters

    samples_per_image = num_correspondences // len(views)
    for view in views:
        correspondences += view.sample_in_mask(samples_per_image)

    for view in views:
        view.nearest_neighbour_match(
            correspondences,
            max_dist=max_dist,
            subpixel_scaling=subpixel_scaling
        )

    return correspondences