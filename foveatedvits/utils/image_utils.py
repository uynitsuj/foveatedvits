import cv2
import numpy as np
from typing import Optional

def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by padding with black.

    Args:
        images: Input image(s) with shape (h, w, c) or (b, h, w, c)
        height: Target height
        width: Target width
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)

    Returns:
        Resized and padded image(s) with shape (height, width, c) or (b, height, width, c)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension

    batch_size, cur_height, cur_width, channels = images.shape

    # Calculate scaling ratio to maintain aspect ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Process each image in the batch
    resized_images = np.zeros((batch_size, resized_height, resized_width, channels), dtype=images.dtype)

    for i in range(batch_size):
        resized_images[i] = cv2.resize(images[i], (resized_width, resized_height), interpolation=interpolation)

    # Calculate padding amounts
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Determine padding value based on dtype
    if images.dtype == np.uint8:
        pad_value = 0
    elif images.dtype == np.float32:
        pad_value = -1.0
    else:
        pad_value = 0

    # Apply padding
    padded_images = np.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    # Remove batch dimension if it wasn't in the input
    if not has_batch_dim:
        padded_images = padded_images[0]

    return padded_images

def resize_with_center_crop(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
    percent_center_crop: Optional[float] = None
) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by center cropping.

    Args:
        images: Input image(s) with shape (h, w, c) or (b, h, w, c)
        height: Target height
        width: Target width
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)
        percent_center_crop: If provided, first crop the center `p`% of each dimension before resizing.

    Returns:
        Resized and center-cropped image(s) with shape (height, width, c) or (b, height, width, c)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension

    batch_size, cur_height, cur_width, channels = images.shape

    # Optionally crop center percent before resizing
    if percent_center_crop is not None:
        assert 0 < percent_center_crop <= 1.0, "percent_center_crop must be in (0, 1]"
        crop_h = int(cur_height * percent_center_crop)
        crop_w = int(cur_width * percent_center_crop)
        crop_h0 = (cur_height - crop_h) // 2
        crop_w0 = (cur_width - crop_w) // 2
        images = images[:, crop_h0:crop_h0+crop_h, crop_w0:crop_w0+crop_w]

        cur_height = crop_h
        cur_width = crop_w

    # Calculate scaling ratio to ensure both dimensions are at least as large as target
    ratio = max(height / cur_height, width / cur_width)
    resized_height = int(cur_height * ratio)
    resized_width = int(cur_width * ratio)

    # Process each image in the batch
    cropped_images = np.zeros((batch_size, height, width, channels), dtype=images.dtype)

    for i in range(batch_size):
        # Resize image so that the smaller dimension fits the target
        resized_img = cv2.resize(images[i], (resized_width, resized_height), interpolation=interpolation)

        # Calculate crop offsets to center the crop
        crop_h0 = (resized_height - height) // 2
        crop_w0 = (resized_width - width) // 2

        cropped_img = resized_img[crop_h0:crop_h0 + height, crop_w0:crop_w0 + width]

        # In rare cases, ensure the output shape is exact
        if cropped_img.shape[0] != height or cropped_img.shape[1] != width:
            cropped_img = cv2.resize(cropped_img, (width, height), interpolation=interpolation)

        cropped_images[i] = cropped_img

    if not has_batch_dim:
        cropped_images = cropped_images[0]

    return cropped_images
