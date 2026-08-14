"""Tests for ultrasound region extraction.

Every image here is synthetic: a bright moving "beamform" blob on black, with
static bright "annotation" text-like marks that must be excluded.
"""

# Standard libraries
from collections import deque

# Non-standard libraries
import cv2
import numpy as np
import pytest

# Custom libraries
from ultraml.core.extract import (
    EmptyMaskError,
    compute_ultrasound_video_mask,
    create_tight_crop,
    extract_ultrasound_image_foreground,
    extract_ultrasound_video_foreground,
    fill_mask,
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
