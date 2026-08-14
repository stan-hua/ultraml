# ultraml: a simple package for ultrasound ML data preprocessing 

## Example Usage
| Before | After | Extracted Background |
|--------|-------|-------|
| ![aorta-0](https://github.com/user-attachments/assets/970f60ad-4d94-45a2-b242-509d43279921) | ![extracted_aorta-0](https://github.com/user-attachments/assets/4c05a588-96be-461d-92bd-8defeb42181d) | ![pocus_atlas-aorta-1-background](https://github.com/user-attachments/assets/307bf44c-0c70-431e-b31f-4bd7419c9954) |

## Installation
1. Pip
```bash
pip install ultraml
```

2. Local Clone
```bash
git clone https://github.com/stan-hua/ultraml.git
cd ultraml
pip install -e .
```

To run the tests:
```bash
pip install pytest
pytest tests/
```


---
## Example Usages
### 1. (File-Level) Pre-process ultrasound video to individual image frames
```python
from ultraml import convert_video_to_frames
video_path = "path/to/video.mp4"
video_save_dir = "path/to/save/frames"
background_save_path = "path/to/save/frames/background.png"
save_paths, background_save_path = convert_video_to_frames(
    path=video_path,
    save_dir=video_save_dir,
    prefix_fname="frame_",
    background_save_path=background_save_path,
    overwrite=True
)
print(f"{len(save_paths)} Video frames saved at: \n\t{'\n\t'.join(save_paths)}")
print(f"Background Saved = {background_save_path is not None}")
```

### 2. (File-Level) Pre-process ultrasound beamform to individual image frames
```python
# If handling DICOMs, please install pydicom with `pip install pydicom`
from ultraml import convert_dicom_to_frames
dicom_path = "path/to/dicom.dcm"
dicom_save_dir = "path/to/save/dicom_frames"
background_save_path = "path/to/save/frames/background.png"
save_paths, background_save_path = convert_dicom_to_frames(
    path=dicom_path,
    save_dir=dicom_save_dir,
    grayscale=True,
    uniform_num_samples=10,
    background_save_path=background_save_path,
    overwrite=True
)
print(f"{len(save_paths)} DICOM frames saved at: \n\t{'\n\t'.join(save_paths)}")
print(f"Background Saved = {background_save_path is not None}")
```

### 3. (Array-Level) Extract beamform from a video (list of images)
```python
from ultraml import extract_ultrasound_video_foreground, convert_img_to_uint8
video_frames_arr = ...        # list of numpy image arrays
ultrasound_foreground, static_mask = extract_ultrasound_video_foreground(
    img_sequence=video_frames_arr,
    apply_filter=True,
    crop=True,
    keep_color=False,         # set True for colour Doppler, see below
)

# Function returns: (i) the video frames with extracted foreground and
#                   (ii) a mask for the static parts of the image
# To get the background, mask out the static parts of any image frame
first_img = video_frames_arr[0]
background_img = convert_img_to_uint8(first_img)
background_img[~static_mask] = 0
```

By default the output is collapsed to grayscale, of shape `(T, H, W)`. Pass
`keep_color=True` to keep the input's channels and get back `(T, H, W, C)` —
needed for colour Doppler, where the flow overlay is the signal. The mask
itself is always decided on luminance, so colour never determines *where* the
beamform is.

You can also tune the mask heuristics; these are forwarded to
`compute_ultrasound_video_mask`:
```python
ultrasound_foreground, static_mask = extract_ultrasound_video_foreground(
    img_sequence=video_frames_arr,
    std_threshold=5,          # per-pixel variation over time to count as moving
    intensity_threshold=15,   # brightness absorbed when connected to a moving pixel
    blur_size=5,              # median blur kernel, odd
)
```

### 4. (Array-Level) Locate the beamform without touching pixels
Use this to store one bounding box per clip and apply it lazily, instead of
materialising every extracted frame.
```python
from ultraml import compute_ultrasound_video_mask
mask, (y_min, y_max, x_min, x_max) = compute_ultrasound_video_mask(
    img_sequence=video_frames_arr,
    apply_filter=True,
)

# `mask` is a boolean (H, W) array, True over the ultrasound region
cropped = video_frames_arr[:, y_min:y_max, x_min:x_max]
```

### 5. (Array-Level) Extract ultrasound from a single image
```python
from ultraml import extract_ultrasound_image_foreground
img_arr = ...               # single numpy image array
ultrasound_foreground, static_mask = extract_ultrasound_image_foreground(
    img=img_arr,
    apply_filter=True,
    crop=True
)

# Function returns: (i) the image frame with extracted foreground and
#                   (ii) a mask for the estimated background of the image
# To get the background, mask out the static parts of any image frame
background_img = convert_img_to_uint8(img_arr)
background_img[~static_mask] = 0
```


---
## When no beamform can be found

The extraction functions raise `EmptyMaskError` rather than returning a blank
frame. An all-zero result is indistinguishable from a legitimately dark scan,
so returning one silently poisons a dataset with blank samples. This happens
when a clip is frozen (no pixel varies across the sequence), when a single
frame is repeated, or when the region has no extent after filtering.

Handle it per clip when running over a cohort:
```python
from ultraml import extract_ultrasound_video_foreground, EmptyMaskError

try:
    foreground, static_mask = extract_ultrasound_video_foreground(video_frames_arr)
except EmptyMaskError as error:
    print(f"Skipping clip: {error}")
```

`EmptyMaskError` subclasses `ValueError`, so existing `except ValueError`
handlers still catch it.
