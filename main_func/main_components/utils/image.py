import cv2
import numpy as np


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      use_origin: bool = True,
                      ) -> np.ndarray:
                      # colormap: int = cv2.COLORMAP_BONE) -> np.ndarray:
    # colormap can be searched through cv2.COLORMAP + _

    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = np.maximum(mask, 0)
    mask = np.minimum(mask, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception(
    #         "The input image should np.float32 in the range [0, 1]")
    # if use_origin:
    #     cam = heatmap + img
    # else:
    #     cam = heatmap
    # cam = cam / np.max(cam)
    # return np.uint8(255 * cam)
    heatmap = np.float32(heatmap)

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    img = (img * 255).astype(np.float32)
    if use_origin:
        cam = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    else:
        cam = heatmap

    return np.uint8(cam)

