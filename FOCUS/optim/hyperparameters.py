import dataclasses
from FOCUS.utils.hyperparameters import Hyperparameters

@dataclasses.dataclass(frozen=True)
class OptimHyperparameters(Hyperparameters):
    """Hyperparameters for the fusion process.

    Args:
        use_uncertainty: Whether to use uncertainty in the fusion process.
        num_samples_per_view: Number of samples to collect for correspondences per view.
        is_world_space: Whether the camera calibration is in world metric space.
    """

    use_uncertainty: bool = True
    num_samples_per_view: int = 100

    is_world_space: bool = True