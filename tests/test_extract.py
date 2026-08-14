"""Tests for ultrasound region extraction.

Every image here is synthetic: a bright moving "beamform" blob on black, with
static bright "annotation" text-like marks that must be excluded.
"""

# Standard libraries
import os
from collections import deque

# Non-standard libraries
import cv2
import numpy as np
import pytest

# Custom libraries
from ultraml.core.extract import (
    EmptyMaskError,
    compute_ultrasound_video_mask,
    convert_dicom_to_frames,
    convert_img_to_uint8,
    convert_video_to_frames,
    create_tight_crop,
    extract_ultrasound_image_foreground,
    extract_ultrasound_video_foreground,
    fill_mask,
    preprocess_and_save_img_array,
)

H, W = 120, 160


def reference_fill_mask(image, mask, intensity_threshold=1):
    """The original per-pixel region grower, kept as the oracle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    filled = mask.copy()
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    queue = deque(np.argwhere(mask > 0))
    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < gray.shape[0] and 0 <= ny < gray.shape[1]:
                if filled[nx, ny] == 0 and gray[nx, ny] > intensity_threshold:
                    filled[nx, ny] = 255
                    queue.append((nx, ny))
    return np.where(filled > 0, 255, 0).astype(np.uint8)


def make_clip(n_frames=8, colour=False, moving=True, annotate=True):
    """A bright blob that drifts, plus static bright marks in the corner."""
    frames = []
    for t in range(n_frames):
        frame = np.zeros((H, W), dtype=np.uint8)
        offset = (t * 3) % 7 if moving else 0
        # The "beamform": a bright rectangle whose texture changes each frame.
        # A frozen clip repeats one frame exactly, intensity included.
        value = 60 + (t * 20) % 150 if moving else 150
        frame[40 + offset:90 + offset, 50:110] = value
        if annotate:
            # Static bright marks, disconnected from the blob -- the banner.
            frame[5:12, 5:40] = 255
            frame[H - 12:H - 5, W - 40:W - 5] = 255
        frames.append(frame)
    clip = np.stack(frames)
    if colour:
        clip = np.repeat(clip[..., None], 3, axis=-1)
    return clip


class TestFillMaskMatchesTheOriginal:
    """The connected-component rewrite must be a drop-in for the pixel queue."""

    @pytest.mark.parametrize("threshold", [1, 15, 60])
    def test_identical_to_the_reference_grower(self, threshold):
        rng = np.random.default_rng(0)
        image = (rng.random((60, 80)) * 255).astype(np.uint8)
        seed = np.zeros((60, 80), dtype=np.uint8)
        seed[30:34, 40:44] = 255
        assert np.array_equal(
            fill_mask(image, seed, threshold),
            reference_fill_mask(image, seed, threshold),
        )

    def test_a_seed_below_the_threshold_is_still_kept(self):
        # The queue version seeded from the mask regardless of intensity.
        image = np.zeros((20, 20), dtype=np.uint8)
        seed = np.zeros((20, 20), dtype=np.uint8)
        seed[10, 10] = 255
        assert fill_mask(image, seed, intensity_threshold=200)[10, 10] == 255

    def test_an_empty_seed_gives_an_empty_mask(self):
        image = np.full((20, 20), 255, dtype=np.uint8)
        empty = np.zeros((20, 20), dtype=np.uint8)
        assert not fill_mask(image, empty).any()

    def test_a_disconnected_bright_region_is_not_absorbed(self):
        # This is the property that keeps a banner out of the mask.
        image = np.zeros((40, 40), dtype=np.uint8)
        image[5:10, 5:10] = 255       # seeded blob
        image[30:35, 30:35] = 255     # separate bright block
        seed = np.zeros((40, 40), dtype=np.uint8)
        seed[6:9, 6:9] = 255
        filled = fill_mask(image, seed, intensity_threshold=15)
        assert filled[5:10, 5:10].all()
        assert not filled[30:35, 30:35].any()


class TestVideoMask:
    """Locating the beamform by what moves."""

    def test_the_static_annotation_is_excluded(self):
        mask, _ = compute_ultrasound_video_mask(make_clip())
        assert not mask[5:12, 5:40].any()
        assert not mask[H - 12:H - 5, W - 40:W - 5].any()

    def test_the_moving_region_is_included(self):
        mask, _ = compute_ultrasound_video_mask(make_clip())
        assert mask[60:80, 60:100].all()

    def test_the_bbox_excludes_the_annotation(self):
        _, (y_min, y_max, x_min, x_max) = compute_ultrasound_video_mask(make_clip())
        assert y_min > 12 and x_min > 40
        assert y_max < H - 12 and x_max < W - 40

    def test_a_frozen_clip_raises_instead_of_returning_black(self):
        # Previously this produced a full-size all-zero clip and no error.
        with pytest.raises(EmptyMaskError, match="No pixel varies"):
            compute_ultrasound_video_mask(make_clip(moving=False))

    def test_a_colour_clip_is_judged_on_luminance(self):
        mono, _ = compute_ultrasound_video_mask(make_clip())
        colour, _ = compute_ultrasound_video_mask(make_clip(colour=True))
        assert np.array_equal(mono, colour)


class TestVideoForeground:
    """The wrapper's output shape and cropping."""

    def test_grayscale_by_default(self):
        out, _ = extract_ultrasound_video_foreground(make_clip(colour=True))
        assert out.ndim == 3

    def test_keep_color_preserves_channels(self):
        # Colour Doppler flow is the signal; collapsing it destroys the study.
        out, _ = extract_ultrasound_video_foreground(
            make_clip(colour=True), keep_color=True
        )
        assert out.ndim == 4 and out.shape[-1] == 3

    def test_the_crop_is_smaller_than_the_frame(self):
        out, _ = extract_ultrasound_video_foreground(make_clip())
        assert out.shape[1] < H and out.shape[2] < W

    def test_not_cropping_keeps_the_frame_size(self):
        out, mask = extract_ultrasound_video_foreground(make_clip(), crop=False)
        assert out.shape[1:] == (H, W)
        assert mask.shape == (H, W)

    def test_the_static_mask_marks_the_annotation(self):
        _, static = extract_ultrasound_video_foreground(make_clip())
        assert static[5:12, 5:40].all()

    def test_frame_count_is_preserved(self):
        out, _ = extract_ultrasound_video_foreground(make_clip(n_frames=11))
        assert out.shape[0] == 11

    def test_a_frozen_clip_raises(self):
        with pytest.raises(EmptyMaskError):
            extract_ultrasound_video_foreground(make_clip(moving=False))


class TestImageForeground:
    """The single-frame path keeps its behaviour, minus the silent failure."""

    def test_a_blank_image_raises_instead_of_returning_black(self):
        with pytest.raises(EmptyMaskError):
            extract_ultrasound_image_foreground(np.zeros((H, W), dtype=np.uint8))

    def test_a_centred_blob_is_found(self):
        img = np.zeros((H, W), dtype=np.uint8)
        img[40:90, 50:110] = 200
        out, _ = extract_ultrasound_image_foreground(img)
        assert out.shape[0] < H and out.shape[1] < W


class TestTightCrop:
    """The bounding box helper."""

    def test_an_empty_image_reports_no_crop(self):
        assert create_tight_crop(np.zeros((10, 10), dtype=np.uint8)) == (
            None,
            None,
            None,
            None,
        )

    def test_a_single_block_is_bounded(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 6:16] = 255
        y_min, y_max, x_min, x_max = create_tight_crop(mask)
        assert (y_min, y_max, x_min, x_max) == (5, 15, 6, 16)


def write_dicom(path, arr):
    """Write a minimal uncompressed ultrasound DICOM holding `arr`.

    A (H, W) array becomes a single-image DICOM with no NumberOfFrames; a
    (T, H, W) array becomes a multiframe one.
    """
    # pydicom is an optional dependency, so skip rather than fail without it
    pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    if arr.ndim == 3:
        ds.NumberOfFrames = arr.shape[0]
    ds.Rows, ds.Columns = arr.shape[-2], arr.shape[-1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.astype(np.uint8).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


class TestConvertDicomToFrames:
    """The DICOM entry point, which was unreachable before it had a test."""

    def test_a_multiframe_dicom_yields_one_file_per_frame(self, tmp_path):
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=8))
        save_dir = tmp_path / "frames"
        paths, background = convert_dicom_to_frames(
            path=dicom_path, save_dir=str(save_dir), prefix_fname="frame_",
        )
        assert len(paths) == 8
        assert all(os.path.exists(p) for p in paths)
        assert background is None

    def test_a_single_image_dicom_does_not_look_for_frame_count(self, tmp_path):
        """The single-image branch used to fall through into NumberOfFrames."""
        dicom_path = write_dicom(tmp_path / "still.dcm", make_clip(n_frames=1)[0])
        save_dir = tmp_path / "frames"
        paths, _ = convert_dicom_to_frames(path=dicom_path, save_dir=str(save_dir))
        assert len(paths) == 1
        assert os.path.exists(paths[0])

    def test_uniform_sampling_takes_the_requested_count(self, tmp_path):
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=12))
        paths, _ = convert_dicom_to_frames(
            path=dicom_path,
            save_dir=str(tmp_path / "frames"),
            uniform_num_samples=4,
        )
        assert len(paths) == 4

    def test_the_prefix_reaches_the_filenames(self, tmp_path):
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=4))
        paths, _ = convert_dicom_to_frames(
            path=dicom_path,
            save_dir=str(tmp_path / "frames"),
            prefix_fname="scan_",
        )
        assert all(os.path.basename(p).startswith("scan_") for p in paths)

    def test_rerunning_returns_the_same_paths_without_redoing_work(self, tmp_path):
        """The skip path returns the same (paths, background) shape as a real run."""
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=5))
        save_dir = str(tmp_path / "frames")
        first, _ = convert_dicom_to_frames(
            path=dicom_path, save_dir=save_dir, prefix_fname="frame_",
        )
        second, _ = convert_dicom_to_frames(
            path=dicom_path, save_dir=save_dir, prefix_fname="frame_",
        )
        assert sorted(second) == sorted(first)

    def test_a_frozen_clip_raises_rather_than_saving_blanks(self, tmp_path):
        frozen = np.repeat(make_clip(n_frames=1), 6, axis=0)
        dicom_path = write_dicom(tmp_path / "frozen.dcm", frozen)
        with pytest.raises(EmptyMaskError):
            convert_dicom_to_frames(
                path=dicom_path, save_dir=str(tmp_path / "frames"),
            )


def write_video(path, clip):
    """Write a clip to an .mp4, skipping the test if no codec is available."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (clip.shape[2], clip.shape[1])
    )
    if not writer.isOpened():
        pytest.skip("No mp4 encoder available in this environment")
    for frame in clip:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return str(path)


class TestUint8Conversion:
    """Scaling, rather than wrapping, whatever bit depth arrives."""

    def test_a_dim_16_bit_scan_is_scaled_not_rejected(self):
        """A UINT16 image whose max is below 256 is still valid UINT16."""
        img = (np.ones((8, 8)) * 200).astype(np.uint16)
        out = convert_img_to_uint8(img)
        assert out.dtype == np.uint8
        assert out.max() == 0  # 200 / 256 rounds down, but does not raise

    def test_a_16_bit_clip_is_scaled_before_masking(self):
        """`.astype(np.uint8)` would wrap these values modulo 256."""
        clip_8bit = make_clip(n_frames=8)
        clip_16bit = (clip_8bit.astype(np.uint16) * 256)
        _, bbox_8bit = compute_ultrasound_video_mask(clip_8bit)
        _, bbox_16bit = compute_ultrasound_video_mask(clip_16bit)
        assert bbox_16bit == bbox_8bit


class TestBackgroundIsAnImage:
    """The background must be saved as a picture, not a bag of pixels."""

    def test_the_saved_background_keeps_its_shape(self, tmp_path):
        img = make_clip(n_frames=1)[0]
        background_path = tmp_path / "nested" / "background.png"
        preprocess_and_save_img_array(
            img,
            extract_beamform=True,
            crop=False,
            background_save_path=str(background_path),
        )
        saved = cv2.imread(str(background_path), cv2.IMREAD_UNCHANGED)
        assert saved.shape[:2] == img.shape[:2]

    def test_a_bare_filename_needs_no_directory(self, tmp_path, monkeypatch):
        """`os.path.dirname` is empty here, and `os.makedirs` rejects that."""
        monkeypatch.chdir(tmp_path)
        preprocess_and_save_img_array(make_clip(n_frames=1)[0], save_path="frame.png")
        assert os.path.exists(tmp_path / "frame.png")


class TestKeepColorOnImages:
    """Single-image extraction mirrors the video function's colour handling."""

    def test_colour_survives_when_asked_for(self):
        img = make_clip(n_frames=1, colour=True)[0]
        out, _ = extract_ultrasound_image_foreground(img, keep_color=True)
        assert out.ndim == 3 and out.shape[2] == 3

    def test_grayscale_remains_the_default(self):
        img = make_clip(n_frames=1, colour=True)[0]
        out, _ = extract_ultrasound_image_foreground(img)
        assert out.ndim == 2


class TestConvertVideoToFrames:
    """The video entry point, including the paths that returned wrong shapes."""

    def test_frames_are_written(self, tmp_path):
        video_path = write_video(tmp_path / "clip.mp4", make_clip(n_frames=6))
        paths, _ = convert_video_to_frames(
            path=video_path, save_dir=str(tmp_path / "frames"), prefix_fname="frame_",
        )
        assert len(paths) == 6
        assert all(os.path.exists(p) for p in paths)

    def test_a_background_alongside_cropping_does_not_raise(self, tmp_path):
        """Frame-level cropping shrinks the frames the mask is computed on."""
        video_path = write_video(tmp_path / "clip.mp4", make_clip(n_frames=6))
        paths, _ = convert_video_to_frames(
            path=video_path,
            save_dir=str(tmp_path / "frames"),
            background_save_path=str(tmp_path / "bg" / "background.png"),
            crop=True,
        )
        assert len(paths) == 6

    def test_rerunning_returns_the_documented_pair(self, tmp_path):
        """The skip path used to return a bare list of filenames."""
        video_path = write_video(tmp_path / "clip.mp4", make_clip(n_frames=6))
        save_dir = str(tmp_path / "frames")
        convert_video_to_frames(
            path=video_path, save_dir=save_dir, prefix_fname="frame_",
        )
        paths, background = convert_video_to_frames(
            path=video_path, save_dir=save_dir, prefix_fname="frame_",
        )
        assert len(paths) == 6
        assert background is None
