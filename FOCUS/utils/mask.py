import numpy as np
import cv2


def largest_island_only(mask: np.ndarray) -> np.ndarray:
    """
    Given a binary mask, return a binary mask of the largest island
    :param mask: binary mask [H x W]
    """
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    area_count = np.bincount(labels.flatten())

    if len(area_count) == 1:
        return mask

    largest_island = np.argmax(area_count[1:]) + 1  # ignore background - '0'
    largest_island_mask = (labels == largest_island)

    return largest_island_mask