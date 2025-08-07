import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union


def resize_and_pad_video(
    input_path: str,
    output_path: str,
    target_size: int = 224,
    target_aspect_ratio: Optional[Tuple[int, int]] = None,
    encoder: str = "h264_nvenc",
    crf: Optional[int] = None,
    bitrate: Optional[str] = None,
    overwrite: bool = True,
    fps: Optional[Union[int, float, str]] = None,
    frame_stride: Optional[int] = None,
    keep_left_half: bool = False,
    crop_to_square: bool = False,
    percent_center_crop: Optional[float] = None
) -> bool:
    """
    Resize, pad, and optionally subsample a video using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        target_size: Target width/height for square output (default: 224)
        target_aspect_ratio: Optional tuple (width, height) for non-square output
        encoder: Video encoder to use (default: "h264_nvenc", fallback: "libx264")
        crf: Constant Rate Factor for quality (0-51, lower is better quality)
        bitrate: Target bitrate (e.g., "2M", "1500k")
        overwrite: Whether to overwrite existing output file
        fps: Target frame rate (e.g., 15, 30, "30000/1001"). If None, keeps original fps
        frame_stride: Take every Nth frame (e.g., stride=2 takes every other frame).
                     Alternative to fps for precise frame subsampling
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If parameters are invalid
    
    Note:
        - If both fps and frame_stride are specified, frame_stride takes precedence
        - frame_stride=2 means take every 2nd frame (halves frame rate)
        - frame_stride=3 means take every 3rd frame (1/3 frame rate)
    """
    
    # Validate inputs
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    
    if frame_stride is not None and frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    
    # Create output directory if it doesn't exist
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle file overwrite
    if output_path_obj.exists() and not overwrite:
        raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
    
    # Determine target dimensions
    if target_aspect_ratio:
        target_width, target_height = target_aspect_ratio
    else:
        target_width = target_height = target_size
    
    # Build ffmpeg command
    cmd = ["ffmpeg"]
    
    # Input
    cmd.extend(["-i", str(input_path)])
    
    # Build video filter chain
    filters = []
    
    # Temporal subsampling filter
    if frame_stride is not None:
        # Use select filter for precise frame selection
        filters.append(f"select='not(mod(n,{frame_stride}))'")
    elif fps is not None:
        # Use fps filter for frame rate conversion
        filters.append(f"fps={fps}")
    
    if keep_left_half:
        filters.append("crop=iw/2:ih:0:0")

    if crop_to_square:
        if percent_center_crop is not None:
            p = percent_center_crop
            filters.append(
                f"crop='min(iw\\,ih)*{p}':'min(iw\\,ih)*{p}':"
                f"'(iw - min(iw\\,ih)*{p})/2':'(ih - min(iw\\,ih)*{p})/2'"
            )        
        else:
            filters.append(
                "crop='min(iw\\,ih)':'min(iw\\,ih)':'(iw-min(iw\\,ih))/2':'(ih-min(iw\\,ih))/2'"
            )
        filters.append(f"scale={target_size}:{target_size}")

    else:
        # Resize preserving aspect ratio, then pad to target dimensions
        if target_aspect_ratio:
            target_width, target_height = target_aspect_ratio
            scale_filter = f"scale='if(gt(a,{target_width}/{target_height}),{target_width},-1)':'if(gt(a,{target_width}/{target_height}),-1,{target_height})'"
        else:
            scale_filter = f"scale='if(gt(a,1),{target_size},-1)':'if(gt(a,1),-1,{target_size})'"
        filters.append(scale_filter)
        filters.append(f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2")


    # Combine all filters
    vf = ",".join(filters)
    cmd.extend(["-vf", vf])
    
    # Video encoding options
    cmd.extend(["-c:v", encoder])
    
    # Quality/bitrate settings
    if crf is not None:
        cmd.extend(["-crf", str(crf)])
    elif bitrate is not None:
        cmd.extend(["-b:v", bitrate])
    
    # If using frame selection, need to handle timestamps
    if frame_stride is not None:
        cmd.extend(["-vsync", "vfr"])  # Variable frame rate to handle selected frames
    
    # Overwrite output file
    if overwrite:
        cmd.append("-y")
    
    # Output
    cmd.append(str(output_path))
    try:
        # Run ffmpeg command on GPU
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        # If NVENC fails, try with CPU encoder as fallback
        if encoder == "h264_nvenc" and "h264_nvenc" in str(e.stderr):
            print(f"NVENC encoder failed, falling back to libx264...")
            return resize_and_pad_video(
                input_path=str(input_path),
                output_path=str(output_path),
                target_size=target_size,
                target_aspect_ratio=target_aspect_ratio,
                encoder="libx264",
                crf=crf,
                bitrate=bitrate,
                overwrite=overwrite,
                fps=fps,
                frame_stride=frame_stride,
                keep_left_half=keep_left_half,
                crop_to_square=crop_to_square,
                percent_center_crop=percent_center_crop
            )
        
        print(f"FFmpeg error: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
