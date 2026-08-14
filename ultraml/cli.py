"""
cli.py

Description: Command-line interface for ultraml, so a cohort can be
             preprocessed without writing any Python.
"""

# Standard libraries
import glob
import os

# Custom libraries
from ultraml.core.extract import (
    EmptyMaskError,
    convert_dicom_to_frames,
    convert_video_to_frames,
)


# Extensions recognised by `batch`, by input kind
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
DICOM_EXTENSIONS = (".dcm", ".dicom")


def video(
        path, save_dir, prefix_fname="",
        background_save_path=None,
        overwrite=False,
        grayscale=None,
        crop=None,
        apply_filter=None,
    ):
    """
    Extract the ultrasound beamform from a video, saving one image per frame.

    Parameters
    ----------
    path : str
        Path to the video file
    save_dir : str
        Directory to save extracted frames to
    prefix_fname : str, optional
        Prefix to prepend to every saved frame
    background_save_path : str, optional
        If provided, save the extracted background to this path
    overwrite : bool, optional
        If True, re-extract even when `save_dir` is already populated
    grayscale : bool, optional
        If True, save frames in grayscale. Left unset, the library's default
        applies.
    crop : bool, optional
        If True, crop frames to the beamform's bounding box
    apply_filter : bool, optional
        If True, median blur the mask to close gaps and drop speckle

    Returns
    -------
    str
        Human-readable summary of what was written
    """
    save_paths, background_save_path = convert_video_to_frames(
        path=path,
        save_dir=save_dir,
        prefix_fname=prefix_fname,
        background_save_path=background_save_path,
        overwrite=overwrite,
        **_processing_kwargs(grayscale, crop, apply_filter),
    )
    return _summarize(save_paths, background_save_path, save_dir)


def dicom(
        path, save_dir, prefix_fname="",
        uniform_num_samples=-1,
        background_save_path=None,
        overwrite=False,
        grayscale=None,
        crop=None,
        apply_filter=None,
    ):
    """
    Extract the ultrasound beamform from a DICOM, saving one image per frame.

    Requires pydicom: `pip install pydicom`

    Parameters
    ----------
    path : str
        Path to the DICOM file
    save_dir : str
        Directory to save extracted frames to
    prefix_fname : str, optional
        Prefix to prepend to every saved frame
    uniform_num_samples : int, optional
        If > 0, evenly sample this many frames from a multiframe DICOM.
        Defaults to -1, which keeps every frame.
    background_save_path : str, optional
        If provided, save the extracted background to this path
    overwrite : bool, optional
        If True, re-extract even when `save_dir` is already populated
    grayscale : bool, optional
        If True, save frames in grayscale. Set False for colour Doppler.
        Left unset, the library's default applies.
    crop : bool, optional
        If True, crop frames to the beamform's bounding box
    apply_filter : bool, optional
        If True, median blur the mask to close gaps and drop speckle

    Returns
    -------
    str
        Human-readable summary of what was written
    """
    save_paths, background_save_path = convert_dicom_to_frames(
        path=path,
        save_dir=save_dir,
        prefix_fname=prefix_fname,
        uniform_num_samples=uniform_num_samples,
        background_save_path=background_save_path,
        overwrite=overwrite,
        **_processing_kwargs(grayscale, crop, apply_filter),
    )
    return _summarize(save_paths, background_save_path, save_dir)


def batch(
        in_dir, save_dir,
        pattern=None,
        uniform_num_samples=-1,
        overwrite=False,
        grayscale=None,
        crop=None,
        apply_filter=None,
    ):
    """
    Extract every video and DICOM under a directory, one sub-directory each.

    A clip with no findable beamform is reported and skipped rather than
    stopping the run, since one frozen clip should not abort a cohort.

    Parameters
    ----------
    in_dir : str
        Directory to search for videos and DICOMs
    save_dir : str
        Directory to save extracted frames to, one sub-directory per input
    pattern : str, optional
        Glob pattern relative to `in_dir`. Defaults to every recognised video
        and DICOM extension, searched recursively.
    uniform_num_samples : int, optional
        If > 0, evenly sample this many frames from each multiframe input
    overwrite : bool, optional
        If True, re-extract even when an output directory is already populated
    grayscale : bool, optional
        If True, save frames in grayscale. Set False for colour Doppler.
    crop : bool, optional
        If True, crop frames to the beamform's bounding box
    apply_filter : bool, optional
        If True, median blur the mask to close gaps and drop speckle

    Returns
    -------
    str
        Human-readable summary of how many inputs succeeded, and why any
        were skipped
    """
    paths = _find_inputs(in_dir, pattern)
    if not paths:
        return f"No videos or DICOMs found under `{in_dir}`"

    num_frames = 0
    converted, skipped = [], []
    for path in paths:
        extension = os.path.splitext(path)[1].lower()
        convert = (
            convert_dicom_to_frames if extension in DICOM_EXTENSIONS
            else convert_video_to_frames
        )
        # One sub-directory per input, named after the file
        curr_save_dir = os.path.join(
            save_dir, os.path.splitext(os.path.basename(path))[0]
        )
        curr_kwargs = _processing_kwargs(grayscale, crop, apply_filter)
        if extension in DICOM_EXTENSIONS:
            curr_kwargs["uniform_num_samples"] = uniform_num_samples

        try:
            save_paths, _ = convert(
                path=path, save_dir=curr_save_dir, overwrite=overwrite,
                **curr_kwargs,
            )
        # NOTE: A clip with no beamform is a data problem, not a run-ending one
        except EmptyMaskError as error_msg:
            skipped.append((path, str(error_msg)))
            continue
        converted.append(path)
        num_frames += len(save_paths)

    summary = [
        f"Converted {len(converted)}/{len(paths)} inputs "
        f"({num_frames} frames) to `{save_dir}`"
    ]
    if skipped:
        summary.append(f"Skipped {len(skipped)}:")
        summary.extend(f"\t{path}\n\t\t{reason}" for path, reason in skipped)
    return "\n".join(summary)


def _processing_kwargs(grayscale=None, crop=None, apply_filter=None):
    """
    Collect the preprocessing flags the user actually set.

    Unset flags are left out entirely rather than passed as their apparent
    default, because these are forwarded to both frame-level and video-level
    preprocessing, whose defaults differ. Passing `crop=True` here would turn
    on frame-level cropping, which the library deliberately leaves off.

    Parameters
    ----------
    grayscale : bool, optional
        If True, save frames in grayscale
    crop : bool, optional
        If True, crop to the beamform's bounding box
    apply_filter : bool, optional
        If True, median blur the mask

    Returns
    -------
    dict
        Keyword arguments to forward, omitting anything left unset
    """
    kwargs = {
        "grayscale": grayscale, "crop": crop, "apply_filter": apply_filter,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _find_inputs(in_dir, pattern=None):
    """
    Find every video and DICOM under a directory.

    Parameters
    ----------
    in_dir : str
        Directory to search
    pattern : str, optional
        Glob pattern relative to `in_dir`. Defaults to every recognised
        extension, searched recursively.

    Returns
    -------
    list of str
        Sorted paths to the inputs found
    """
    assert os.path.isdir(in_dir), f"Input directory does not exist! \n\tPath: {in_dir}"

    if pattern:
        return sorted(glob.glob(os.path.join(in_dir, pattern), recursive=True))

    paths = []
    for extension in VIDEO_EXTENSIONS + DICOM_EXTENSIONS:
        paths.extend(
            glob.glob(os.path.join(in_dir, "**", f"*{extension}"), recursive=True)
        )
    return sorted(paths)


def _summarize(save_paths, background_save_path, save_dir):
    """
    Describe what a conversion wrote.

    Parameters
    ----------
    save_paths : list of str
        Paths to the saved image frames
    background_save_path : str or None
        Path to the saved background image, if one was saved
    save_dir : str
        Directory the frames were saved to

    Returns
    -------
    str
        Human-readable summary
    """
    summary = [f"Saved {len(save_paths)} frames to `{save_dir}`"]
    if background_save_path:
        summary.append(f"Saved background to `{background_save_path}`")
    return "\n".join(summary)


def main():
    """
    Entry point for the `ultraml` command.
    """
    # Lazy import, so that importing the library does not pay for the CLI
    import fire

    fire.Fire({
        "video": video,
        "dicom": dicom,
        "batch": batch,
    })


if __name__ == "__main__":
    main()
