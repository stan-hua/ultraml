"""
extract.py

Description: Contains helper functions to extract ultrasound beamforms from
             the background.
"""

# Standard libraries
import logging
import os
from collections import deque
from joblib import Parallel, delayed

# Non-standard libraries
import cv2
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm


################################################################################
#                               Helper Functions                               #
################################################################################
def convert_video_to_frames(
        path, save_dir, prefix_fname="",
        background_save_path=None,
        overwrite=False,
        **kwargs,
    ):
    """
    Convert video to image frames

    Parameters
    ----------
    path : str
        Path to video
    save_dir : str
        Path to directory to save video
    prefix_fname : str
        Prefix to prepend to all image frames
    background_save_path : str
        If provided, save extracted background to file
    overwrite : bool
        If True, overwrite existing frames, by default False. Otherwise, simply
        return filenames
    **kwargs : Any
        Keyword arguments for frame-level preprocessing
        `preprocess_and_save_img_array`, which includes
        `grayscale`, `extract_beamform`, `crop`, and `apply_filter`. By default,
        the beamform is extracted at the frame-level first to remove colored
        components and potential noise that is static across frames.

    Returns
    -------
    tuple of (list of str, str)
        (i) Paths to saved image frames
        (ii) Path to saved background image, or None if not saved
    """
    #  Default keyword arguments for processing images
    process_kwargs = {
        "grayscale": False,
        "extract_beamform": True,
        "crop": False,
        "apply_filter": False,
    }
    process_kwargs.update(kwargs)

    # Local function to process image frame
    def process_frame(idx, img_arr):
        curr_save_path = f"{save_dir}/{prefix_fname}{idx}.png"
        processed_image = preprocess_and_save_img_array(img_arr, **process_kwargs)
        return curr_save_path, processed_image

    os.makedirs(save_dir, exist_ok=True)
    # Simply return filenames if already exists
    if not overwrite and os.listdir(save_dir):
        print("[Video Conversion] Already exists, skipping...")
        num_files = len(os.listdir(save_dir))
        # Recreate filenames
        idx = 1
        paths = [f"{prefix_fname}{idx+i}.png" for i in range(num_files)]
        assert set(paths) == set(os.listdir(save_dir)), (
            f"Unexpected error! Previously extracted video frames have "
            "unexpected file names. Please delete `{save_dir}`"
        )
        return paths

    # Convert video to frames
    vidcap = cv2.VideoCapture(path)
    success, img_arr = vidcap.read()
    idx = 1
    frames_args = []
    while success:
        frames_args.append((idx, img_arr))
        # Load next image
        success, img_arr = vidcap.read()
        idx += 1

    # Process images in parallel
    ordered_results = Parallel(n_jobs=-1)(
        delayed(process_frame)(idx, img_arr)
        for idx, img_arr in frames_args
    )

    # Get processed images and their save paths
    accum_imgs = []
    img_save_paths = []
    for curr_img_path, curr_img in ordered_results:
        accum_imgs.append(curr_img)
        img_save_paths.append(curr_img_path)

    # Early return, if no images extracted
    if not accum_imgs:
        return []

    # Separate out ultrasound & non-ultrasound part of sequence
    # CASE 1: Only 1 image frame
    if len(accum_imgs) == 1:
        foreground, static_mask = extract_ultrasound_image_foreground(accum_imgs[0])
        foreground = [foreground]
    # CASE 2: Video
    else:
        foreground, static_mask = extract_ultrasound_video_foreground(np.array(accum_imgs))

    # Save extracted ultrasound part to save paths
    for image_idx, save_img_path in enumerate(img_save_paths):
        cv2.imwrite(save_img_path, foreground[image_idx])

    # If specified, extract background image from first image
    if background_save_path and static_mask is not None:
        first_img = frames_args[0][1]
        background_img = convert_img_to_uint8(first_img)
        background_img[~static_mask] = 0
        # NOTE: Only save if background image has at least 25 non-zero pixels
        if background_img.sum() >= 25:
            os.makedirs(os.path.dirname(background_save_path), exist_ok=True)
            cv2.imwrite(background_save_path, background_img)

    return img_save_paths, background_save_path


def convert_dicom_to_frames(
        path, save_dir, prefix_fname="",
        uniform_num_samples=-1,
        background_save_path=None,
        bg_min_pixels=25,
        overwrite=False,
        **kwargs,
    ):
    """
    Convert DICOM image/video to 1+ image frames

    Parameters
    ----------
    path : str
        Path to video
    save_dir : str
        Path to directory to save video
    prefix_fname : str
        Prefix to prepend to all image frames
    uniform_num_samples : int, optional
        If DICOM contains a video and this value is > 0, sample uniformly across
        the number of frames in the video.
    background_save_path : str
        If provided, save extracted background to file
    bg_min_pixels : int
        Minimum number of non-zero pixels in background image to save it
    overwrite : bool
        If True, overwrite existing frames, by default False. Otherwise, simply
        return filenames
    **kwargs : Any
        Keyword arguments for `preprocess_and_save_img_array`, which includes
        `grayscale`, `extract_beamform`, `crop`, and `apply_filter`

    Returns
    -------
    tuple of (list of str, str)
        (i) Paths to saved image frames
        (ii) Path to saved background image, or None if not saved
    """
    #  Default keyword arguments for processing images
    process_kwargs = {
        "grayscale": True,
    }
    process_kwargs.update(kwargs)

    # Lazy import to speed up file loading
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is not installed. Please install it using `pip install pydicom`"
        )
    os.makedirs(save_dir, exist_ok=True)

    # Simply return filenames if already exists
    if not overwrite and os.listdir(save_dir):
        print("[DICOM Conversion] Already exists, skipping...")
        exist_paths = os.listdir(save_dir)
        num_files = len(exist_paths)
        # Recreate filenames
        paths = [f"{prefix_fname}{1+i}.png" for i in range(num_files)]
        assert set(paths) == set(exist_paths), (
            "Unexpected error! Previously extracted video frames have "
            f"unexpected file names. Please delete `{save_dir}`"
        )
        return paths

    # Load DICOM
    assert os.path.exists(path), f"DICOM does not exist at path! \n\tPath: {path}"
    dicom_obj = pydicom.dcmread(path)

    # CASE 1: A single image
    if not hasattr(dicom_obj, "NumberOfFrames"):
        img_arr = dicom_obj.pixel_array

        # Preprocess image and save to path
        preprocess_and_save_img_array(
            img_arr, grayscale,
            save_path=f"{save_dir}/{prefix_fname}1.png",
            background_save_path=background_save_path,
        )

    # CASE 2: A sequence of image frames
    num_frames = int(dicom_obj.NumberOfFrames)

    # Get frame indices based on sampling choice
    img_indices = list(range(num_frames))
    # 1. Deterministically sample uniformly across the sequence
    if uniform_num_samples > 0:
        if uniform_num_samples > num_frames:
            print("Cannot sample more frames than available! Defaulting to all frames...")
        # Uniformly sampling
        else:
            img_indices = list(np.linspace(0, num_frames-1, uniform_num_samples, dtype=int))

    # Get all raw image frames and their future save paths
    accum_imgs = [dicom_obj.pixel_array[arr_idx] for arr_idx in img_indices]
    img_save_paths = [f"{save_dir}/{prefix_fname}{idx+1}.png" for idx in range(len(accum_imgs))]

    # Separate out ultrasound & non-ultrasound part of sequence
    # CASE 1: Only 1 image frame
    if len(accum_imgs) == 1:
        foreground, static_mask = extract_ultrasound_image_foreground(accum_imgs[0])
        foreground = [foreground]
    # CASE 2: Video
    else:
        foreground, static_mask = extract_ultrasound_video_foreground(np.array(accum_imgs))

    # Save extracted ultrasound part to save paths
    for image_idx, save_img_path in enumerate(img_save_paths):
        cv2.imwrite(save_img_path, foreground[image_idx])

    # If specified, extract background image from first image
    if background_save_path and static_mask is not None:
        first_img = frames_args[0][1]
        background_img = convert_img_to_uint8(first_img)
        background_img[~static_mask] = 0
        # NOTE: Only save if background image has at least 25 non-zero pixels
        if background_img.sum() >= bg_min_pixels:
            os.makedirs(os.path.dirname(background_save_path), exist_ok=True)
            cv2.imwrite(background_save_path, background_img)
        else:
            background_save_path = None

    return img_save_paths, background_save_path


def convert_img_to_uint8(img_arr):
    """
    Convert images (e.g., UINT16) to UINT8.

    Parameters
    ----------
    img_arr : np.ndarray
        Image array to be converted.

    Returns
    -------
    img_arr : np.ndarray
        Converted image array.
    """
    # CASE 0: Image is already UINT8
    if img_arr.dtype == np.uint8:
        return img_arr

    # CASE 1: If image is UINT16, convert to UINT8 by dividing by 256
    if img_arr.dtype == np.uint16:
        img_arr = img_arr.astype(np.float32)
        assert img_arr.max() > 255, f"[UINT16 to UINT8] Image has pixel value > 255! Max: {img_arr.max()}"
        return np.clip((img_arr / 256), 0, 255).astype(np.uint8)
    # CASE 2: If image is between 0 and 1, then multiply by 255
    elif img_arr.min() >= 0 and img_arr.max() <= 1:
        return np.clip((img_arr * 255), 0, 255).astype(np.uint8)

    # Raise error with unhandled dtype
    raise NotImplementedError(f"[UINT16 to UINT8] Unsupported image type! dtype: `{img_arr.dtype}`")


def preprocess_and_save_img_array(
        img_arr, grayscale=True, extract_beamform=False,
        save_path=None, background_save_path=None,
        **kwargs,
    ):
    """
    Preprocess the input image array by converting it to UINT8 and optionally 
    ensuring it is in grayscale format.

    Parameters
    ----------
    img_arr : np.ndarray
        Image array to be preprocessed. Expected to be in either UINT16 or 
        RGB format if conversion is necessary.
    grayscale : bool, optional
        If True, ensures the output image is in grayscale format. Default is True.
    extract_beamform : bool, optional
        If True, extract ultrasound beamform part from image. Default is False.
    save_path : str, optional
        Path to save the preprocessed image array. Default is None.
    background_save_path : str, optional
        If `extract_beamform` and provided, save the non-ultrasound part of the image.
        Default is None.
    **kwargs : Keyword arguments
        Additional keyword arguments to be passed to `extract_ultrasound_image_foreground`

    Returns
    -------
    np.ndarray
        The processed image array in UINT8 format and optionally in grayscale.
    """
    # Preprocess image
    # 1. Convert to UINT8
    img_arr = convert_img_to_uint8(img_arr)

    # If specified, extract beamform part of ultrasound image
    if extract_beamform:
        img_arr, static_mask = extract_ultrasound_image_foreground(img_arr, **kwargs)
        # Save background, if specified
        if background_save_path:
            cv2.imwrite(background_save_path, img_arr[static_mask])

    # 2. Ensure grayscale image, if specified
    if grayscale and len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    # Save image to file, if specified
    if save_path:
        cv2.imwrite(save_path, img_arr)

    return img_arr


def is_image_dark(img_arr):
    """
    Return True if image is more than 75% of the image is dark/black pixels,
    and False otherwise

    Parameters
    ----------
    img_arr : np.array
        Image array with pixel values in [0, 255]

    Returns
    -------
    bool
        True if image is dark, False otherwise
    """
    # Convert to grayscale if not already
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    # Checks if more than 70% of the image is dark pixels
    return np.mean(img_arr < 30) >= 0.60


def extract_ultrasound_video_foreground(img_sequence, apply_filter=True, crop=True):
    """
    Split ultrasound video into foreground (ultrasound beamform) and background
    (unecessary static parts).

    Parameters
    ----------
    img_sequence : np.ndarray
        Image sequence to separate foreground from background. Of shape (T, H, W, C)
    apply_filter : bool, optional
        If True, apply median blur filter to image
    crop : bool, optional
        If True, return cropped image

    Returns
    -------
    (np.ndarray, np.ndarray)
        (i) Ultrasound video with beamform extracted of shape (T, H, W), where
            H and W can be smaller due to cropping
        (ii) Boolean mask of shape (H, W) that highlights static parts of video
             frames that denote the background
    """
    img_sequence = img_sequence.astype(np.uint8)

    # Convert to grayscale
    if len(img_sequence.shape) == 4 and img_sequence.shape[3] == 3:
        grayscale_imgs = []
        for idx in range(len(img_sequence)):
            grayscale_imgs.append(cv2.cvtColor(img_sequence[idx], cv2.COLOR_RGB2GRAY))
        img_sequence = np.stack(grayscale_imgs, axis=0)

    # Create mask of shape (H, W) that indicates parts of image with no variation
    dynamic_mask = (np.std(img_sequence, axis=0) >= 5)
    dynamic_mask = (255*dynamic_mask).astype(np.uint8)

    # Use maximum pixel intensity to fill in the mask
    # NOTE: Bright pixels by the mask should be included
    max_img = img_sequence.max(0)
    dynamic_mask = fill_mask(max_img, dynamic_mask, intensity_threshold=15)

    # If specified, use median blur filter to fill in the gaps and remove noise
    if apply_filter:
        dynamic_mask = cv2.medianBlur(dynamic_mask, 5)
        # Convert back to binary mask
        dynamic_mask = (255*(dynamic_mask > 0)).astype(np.uint8)

    # Split ultrasound video into ultrasound video and non-ultrasound image
    ultrasound_part, non_ultrasound_part = img_sequence.copy(), img_sequence.copy()
    ultrasound_part[:, ~dynamic_mask.astype(bool)] = 0

    # Extract static part of video
    static_mask = ~dynamic_mask.astype(bool)

    # Early return, if not cropping
    if not crop:
        return ultrasound_part, static_mask

    # Get tightest crop of ultrasound image
    y_min, y_max, x_min, x_max = create_tight_crop(dynamic_mask)
    ultrasound_part = ultrasound_part[:, y_min:y_max, x_min:x_max]

    return ultrasound_part, static_mask


def extract_ultrasound_image_foreground(img, apply_filter=True, crop=True):
    """
    Split ultrasound image into ultrasound (beamform) and non-ultrasound (unecessary static parts).

    Parameters
    ----------
    img : np.ndarray
        Ultrasound image to separate ultrasound from non-ultrasound part.
        Of shape (H, W, C) or (H, W)
    apply_filter : bool, optional
        If True, apply median blur filter to initial mask to remove noise
    crop : bool, optional
        If True, return cropped image

    Returns
    -------
    (np.ndarray, np.ndarray)
        (i) Ultrasound image with beamform extracted of shape (H, W), where
            H and W can be smaller due to cropping
        (ii) Boolean mask of shape (H, W) that highlights static parts of the
             image that denote the background
    """
    # Get 10 points within 30% of the width at the center of the image
    width = img.shape[1]
    middle_idx = int(width * 0.5)
    width_range = int(width * 0.15)
    lower, upper = max(0, middle_idx - width_range), min(width, middle_idx + width_range)
    indices = sorted(np.linspace(lower, upper, 10, dtype=int))

    # Convert to grayscale and get is colored mask for center column of image
    is_colored_center_mask = np.zeros_like(img[:, indices], dtype=bool)
    if len(img.shape) == 3 and img.shape[2] == 3:
        is_colored_center_mask = (np.std(img[:, indices], axis=2) >= 5)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # For center pixels that are greater than 50 and not colored, assume it's part of the mask
    active_mask = np.zeros_like(img, dtype=bool)
    active_mask[:, indices] = (img[:, indices] >= 30) & (~is_colored_center_mask)

    # From the center-filled mask, fill in the remaining part of mask
    active_mask = fill_mask(img, active_mask, intensity_threshold=15)

    # If specified, use median blur filter to fill in the gaps and remove noise
    if apply_filter:
        active_mask = cv2.medianBlur(active_mask, 5)
        # Convert back to binary mask
        active_mask = (active_mask > 0)

    # Split ultrasound image into ultrasound part and non-ultrasound part
    active_mask_bool = active_mask.astype(bool)
    ultrasound_part, non_ultrasound_part = img.copy(), img.copy()
    ultrasound_part[~active_mask_bool] = 0
    static_mask = ~active_mask_bool

    # Early return, if not cropping
    if not crop:
        return ultrasound_part, static_mask

    # Get tightest crop of ultrasound image
    y_min, y_max, x_min, x_max = create_tight_crop(active_mask)
    ultrasound_part = ultrasound_part[y_min:y_max, x_min:x_max]

    return ultrasound_part, static_mask


def fill_mask(image, mask, intensity_threshold=1):
    """
    Fill the mask by using the pixel intensity values that are greater than the threshold.

    Parameters
    ----------
    image : np.ndarray
        The ultrasound image.
    mask : np.ndarray
        The incomplete mask.
    intensity_threshold : int, optional
        The intensity threshold to consider pixels as part of the mask.

    Returns
    -------
    np.ndarray
        The filled mask.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # Create a copy of the mask to update
    filled_mask = mask.copy()

    # Get the coordinates of the initial mask pixels
    initial_points = np.argwhere(mask > 0)

    # Define the 8-connected neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Use a deque for efficient queue operations
    queue = deque(initial_points)

    # Region growing algorithm
    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < gray_image.shape[0] and 0 <= ny < gray_image.shape[1]:
                if filled_mask[nx, ny] == 0 and gray_image[nx, ny] > intensity_threshold:
                    filled_mask[nx, ny] = 255
                    queue.append((nx, ny))

    # Ensure mask is 0 or 255
    filled_mask = np.where(filled_mask > 0, 255, 0).astype(np.uint8)
    return filled_mask


def create_tight_crop(image):
    """
    Get tightest crop that doesn't remove any non-zero pixels

    Parameters
    ----------
    image : np.ndarray
        The image to be cropped.

    Returns
    -------
    tuple of int
        Coordinates to get the tightest crop (y_min, y_max, x_min, x_max)
        Returns (None, None, None, None) if image has no active pixel
    """
    # Find the coordinates of non-zero pixels
    non_zero_coords = np.argwhere(image > 0)

    # Early return, if image is empty
    if len(non_zero_coords) == 0:
        return None, None, None, None

    # Get the bounding box of the non-zero pixels
    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0)
    # Crop the image using the bounding box coordinates
    y_min, y_max, x_min, x_max = top_left[0], bottom_right[0]+1, top_left[1], bottom_right[1]+1
    return y_min, y_max, x_min, x_max


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    from fire import Fire
    Fire()
