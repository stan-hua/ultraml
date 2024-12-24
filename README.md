# ultraml: a package to simplify ultrasound ML data pre-processing


## Installation
1. Pip
```
pip install ultraml
```

2. Local Clone
```
git clone https://github.com/stan-hua/ultraml.git
cd ultraml
pip install -e .
```


---
## Example Usages
### 1. (File-Level) Pre-process ultrasound video to individual image frames
```
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
```
# If handling DICOMs, please install pydicom with `pip install pydicom`
from ultraml import convert_dicom_to_frames
dicom_path = "path/to/dicom.dcm"
dicom_save_dir = "path/to/save/dicom_frames"
background_save_path = "path/to/save/frames/background.png"
save_paths, background_save_path = convert_dicom_to_frames(
    path=dicom_path,
    save_dir=dicom_save_dir,
    prefix_fname="dicom_frame_",
    grayscale=True,
    uniform_num_samples=10,
    background_save_path=background_save_path,
    overwrite=True
)
print(f"{len(save_paths)} DICOM frames saved at: \n\t{'\n\t'.join(save_paths)}")
print(f"Background Saved = {background_save_path is not None}")
```

### 3. (Array-Level) Extract beamform from a video (list of images)
```
from ultraml import extract_ultrasound_video_foreground, convert_img_to_uint8
video_frames_arr = ...        # list of numpy image arrays
ultrasound_foreground, static_mask = extract_ultrasound_video_foreground(
    img_sequence=video_frames_arr,
    apply_filter=True,
    crop=True
)

# Function returns: (i) the video frames with extracted foreground and
#                   (ii) a mask for the static parts of the image
# To get the background, mask out the static parts of any image frame
first_img = video_frames_arr[0]
background_img = convert_img_to_uint8(first_img)
background_img[~static_mask] = 0
```

### 4. (Array-Level) Extract ultrasound from a single image
```
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
