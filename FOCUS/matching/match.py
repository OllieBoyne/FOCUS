"""Find matches across Views."""

from FOCUS.matching import view
from FOCUS.matching import correspondence
import torch

def find_matches(views: [view.View], samples_per_image: int,
                 max_dist: float = 0.002,
                 subpixel_scaling: int = 8
                 ) -> [correspondence.Correspondence]:
    """Find matches across Views."""

    correspondences: list[correspondence.Correspondence] = []

    torch.manual_seed(10) # TODO: add seed to hyperparameters

    for view in views:
        correspondences += view.sample_in_mask(samples_per_image)

    for view in views:
        view.nearest_neighbour_match(
            correspondences,
            max_dist=max_dist,
            subpixel_scaling=subpixel_scaling
        )

    return correspondences