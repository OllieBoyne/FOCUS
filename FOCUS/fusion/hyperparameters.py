import dataclasses

@dataclasses.dataclass(frozen=True)
class FusionHyperparameters:
    """Hyperparameters for the fusion process.

    Args:
        num_correspondences: Number of samples to collect for correspondences across all images.
        toc_correspondence_threshold: Maximum L2 distance between two TOC values to be considered a valid correspondence.
        toc_uncertainty_threshold: Discard correspondences with average TOC uncertainty above this value.
        toc_height_limit: Discard correspondences with a TOC z-value above this value.
        correspondence_reprojection_threshold_pixels: Discard correspondences with a reprojection error above this value.
        bbox_cropping_padding: Crop the mesh to a bounding box containing the original points, with this padding.
        mesh_cutoff_heights: Cutoff the mesh above and below these heights.
        screened_poisson_use_calculated_normals: Use normals calculated from the point cloud for the screened Poisson reconstruction.
        screened_poisson_depth: Depth of the screened Poisson reconstruction.
        screened_poisson_samplespernode: Samples per node for the screened Poisson reconstruction.
        screened_poisson_iters: Number of iterations for the screened Poisson reconstruction.
        match_method: Matching method to use, either "NN" or "Guided".
        nn_upsampling_factor: Upsampling factor for input image to nearest neighbour matching.
        is_world_space: Whether the camera calibration is in world metric space.

    """

    num_correspondences: int = 20_000
    toc_correspondence_threshold: float = 0.002
    toc_uncertainty_threshold: float = 100.0
    toc_height_limit: float = 1.0
    correspondence_reprojection_threshold_pixels: float = 3.0
    bbox_cropping_padding: float = 0.001
    mesh_cutoff_heights: tuple[float, float] = (0.0, 0.15)

    screened_poisson_use_calculated_normals: bool = True
    screened_poisson_depth: int = 8
    screened_poisson_samplespernode: float = 1.5
    screened_poisson_iters: int = 8
    screened_poisson_cgdepth: int = 0
    screened_poisson_pointweight: int = 0

    nn_upsampling_factor: int = 8

    is_world_space: bool = False

    @property
    def screened_poisson_parameters(self):
        return {
            "depth": self.screened_poisson_depth,
            "samplespernode": self.screened_poisson_samplespernode,
            "iters": self.screened_poisson_iters,
            "cgdepth": self.screened_poisson_cgdepth,
            "pointweight": self.screened_poisson_pointweight,
        }