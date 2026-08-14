"""Tests for the `ultraml` command-line interface."""

# Standard libraries
import os

# Non-standard libraries
import numpy as np
import pytest

# Custom libraries
from ultraml import cli
from test_extract import make_clip, write_dicom, write_video


class TestVideoCommand:
    """`ultraml video`"""

    def test_it_reports_what_it_saved(self, tmp_path):
        video_path = write_video(tmp_path / "clip.mp4", make_clip(n_frames=6))
        save_dir = str(tmp_path / "frames")
        summary = cli.video(path=video_path, save_dir=save_dir)
        assert "6 frames" in summary
        assert len(os.listdir(save_dir)) == 6

    def test_a_background_is_reported_when_saved(self, tmp_path):
        video_path = write_video(tmp_path / "clip.mp4", make_clip(n_frames=6))
        summary = cli.video(
            path=video_path,
            save_dir=str(tmp_path / "frames"),
            background_save_path=str(tmp_path / "bg" / "background.png"),
        )
        assert "background" in summary.lower()


class TestDicomCommand:
    """`ultraml dicom`"""

    def test_it_reports_what_it_saved(self, tmp_path):
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=8))
        summary = cli.dicom(path=dicom_path, save_dir=str(tmp_path / "frames"))
        assert "8 frames" in summary

    def test_uniform_sampling_is_honoured(self, tmp_path):
        dicom_path = write_dicom(tmp_path / "clip.dcm", make_clip(n_frames=12))
        summary = cli.dicom(
            path=dicom_path,
            save_dir=str(tmp_path / "frames"),
            uniform_num_samples=4,
        )
        assert "4 frames" in summary


class TestBatchCommand:
    """`ultraml batch`"""

    def test_it_converts_every_input_it_finds(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_dicom(in_dir / "a.dcm", make_clip(n_frames=6))
        write_video(in_dir / "b.mp4", make_clip(n_frames=6))
        summary = cli.batch(in_dir=str(in_dir), save_dir=str(tmp_path / "out"))
        assert "Converted 2/2" in summary
        # One sub-directory per input, named after the file
        assert sorted(os.listdir(tmp_path / "out")) == ["a", "b"]

    def test_one_unusable_clip_does_not_abort_the_run(self, tmp_path):
        """A frozen clip is a data problem, not a reason to stop a cohort."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_dicom(in_dir / "good.dcm", make_clip(n_frames=6))
        write_dicom(in_dir / "frozen.dcm", np.repeat(make_clip(n_frames=1), 6, axis=0))
        summary = cli.batch(in_dir=str(in_dir), save_dir=str(tmp_path / "out"))
        assert "Converted 1/2" in summary
        assert "Skipped 1" in summary
        assert "frozen.dcm" in summary

    def test_an_empty_directory_says_so(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        summary = cli.batch(in_dir=str(in_dir), save_dir=str(tmp_path / "out"))
        assert "No videos or DICOMs found" in summary

    def test_a_missing_directory_fails_loudly(self, tmp_path):
        with pytest.raises(AssertionError):
            cli.batch(in_dir=str(tmp_path / "nope"), save_dir=str(tmp_path / "out"))


class TestProcessingKwargs:
    """Unset flags must not override the library's own defaults."""

    def test_unset_flags_are_dropped(self):
        assert cli._processing_kwargs() == {}

    def test_set_flags_are_forwarded(self):
        assert cli._processing_kwargs(grayscale=False, crop=True) == {
            "grayscale": False,
            "crop": True,
        }
