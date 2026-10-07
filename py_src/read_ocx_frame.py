"""Read selected frames from a Bioptigen OCT/OCU (OCX) file.

Python port of read_OCX_frame.m (original by Alex Salmon, Bioptigen/Brad Bower). Made by Claude Sonnet 5.5 Medium

Requires Python >= 3.12 and numpy.

Conventions
-----------
* Frame indices are 0-based (Python native).
* All binary values are read using the machine's native byte order
  (``struct`` "=" prefix / numpy native dtypes), as MATLAB's fread does.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

# Header keys grouped by the type of value they store
_UINT32_KEYS = {
    "FRAMECOUNT": "frameCount",
    "LINECOUNT": "lineCount",
    "LINELENGTH": "lineLength",
    "SAMPLEFORMAT": "sampleFormat",
    "SCANTYPE": "scanType",
    "SCANS": "scans",
    "FRAMES": "frames",
    "FRAMESPERVOLUME": "framesPerVolume",
    "DOPPLERFLAG": "dopplerFlag",
}
_DOUBLE_KEYS = {
    "XMIN": "xMin",
    "XMAX": "xMax",
    "YMIN": "yMin",
    "YMAX": "yMax",
    "SCANDEPTH": "scanDepth",
    "SCANLENGTH": "scanLength",
    "AZSCANLENGTH": "azScanLength",
    "ELSCANLENGTH": "elScanLength",
    "OBJECTDISTANCE": "objectDistance",
    "SCANANGLE": "scanAngle",
}
_STRING_KEYS = {
    "DESCRIPTION": "description",
    "XCAPTION": "xCaption",
    "YCAPTION": "yCaption",
}

# Constants for per-frame byte counts
KEY_READ = 17    # Bytes for reading key
FRAME_DT = 37    # Bytes for reading frame date and time
FRAME_TS = 30    # Bytes for reading time stamp
FRAME_L = 22     # Bytes for reading frame lines
FRAME_MD = 20    # Bytes for reading frame sample metadata
END_FRAME = 4    # Bytes for reading end of frame
META_BYTES = KEY_READ + FRAME_DT + FRAME_TS + FRAME_L + FRAME_MD


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("Unexpected end of file while reading OCX header")
    return data


# "=" -> native byte order, standard sizes
def _read_u16(f: BinaryIO) -> int:
    return struct.unpack("=H", _read_exact(f, 2))[0]


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("=I", _read_exact(f, 4))[0]


def _read_f64(f: BinaryIO) -> float:
    return struct.unpack("=d", _read_exact(f, 8))[0]


def read_ocx_header(f: BinaryIO) -> tuple[dict[str, Any], int]:
    """Parse the file header.

    Returns the header dict and the file header length in bytes
    (same value as ``fileHeaderLength`` in the MATLAB version).
    """
    head: dict[str, Any] = {}

    head["magicNumber"] = [f"{_read_u16(f):X}" for _ in range(2)]
    head["versionNumber"] = f"{_read_u16(f):X}"

    key_length = _read_u32(f)
    key = _read_exact(f, key_length).decode("ascii", errors="replace")
    if key != "FRAMEHEADER":
        raise ValueError("Error loading frame header")

    _read_u32(f)  # skip storing initial data length

    while True:
        key_length = _read_u32(f)
        key = _read_exact(f, key_length).decode("ascii", errors="replace")
        data_length = _read_u32(f)

        if key in _UINT32_KEYS:
            head[_UINT32_KEYS[key]] = _read_u32(f)
        elif key in _DOUBLE_KEYS:
            head[_DOUBLE_KEYS[key]] = _read_f64(f)
        elif key in _STRING_KEYS:
            head[_STRING_KEYS[key]] = _read_exact(f, data_length).decode(
                "ascii", errors="replace"
            )
        elif key == "CONFIG":
            head["config"] = np.frombuffer(_read_exact(f, data_length), dtype=np.uint8)
        else:
            break  # all header keys read

    if head.get("scanType") == 6:  # mixed mode volume
        raise NotImplementedError("Mixed Density ('M') Scans Not Supported.")

    # Correct for 4-byte keyLength read in frame header loop
    f.seek(-4, 1)
    file_header_length = f.tell()
    return head, file_header_length


def read_ocx_frame(
    ffname: str | Path,
    frame_indices: Iterable[int] | int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the requested frames from an OCX file.

    Parameters
    ----------
    ffname : path to the .oct / .ocu file.
    frame_indices : 0-based frame number(s) to read.

    Returns
    -------
    ocx_frames : uint16 array of shape (lineLength, lineCount, n_frames),
        the same layout as the MATLAB output.
    ocx_head : dict of header fields (same names as the MATLAB struct).
    """
    if isinstance(frame_indices, int):
        frame_indices = [frame_indices]
    indices = list(frame_indices)

    with open(ffname, "rb") as f:
        head, file_header_length = read_ocx_header(f)

        line_length = head["lineLength"]
        line_count = head["lineCount"]
        n_samples = line_length * line_count
        bytes_per_frame = n_samples * 2  # 16-bit (2 bytes)

        frames = np.zeros((line_length, line_count, len(indices)), dtype=np.uint16)

        for ii, fii in enumerate(indices):
            if fii < 0:
                raise IndexError(f"Frame index must be >= 0; got {fii}")
            # Each frame is preceded by META_BYTES of metadata, and followed
            # by END_FRAME bytes; fii earlier frames are fully skipped.
            start_pos = (
                file_header_length
                + (fii + 1) * META_BYTES
                + fii * (bytes_per_frame + END_FRAME)
            )
            f.seek(start_pos)
            raw = np.fromfile(f, dtype=np.uint16, count=n_samples)  # native order
            if raw.size != n_samples:
                raise EOFError(f"Frame {fii} is truncated or out of range")
            # MATLAB fread fills column-major -> Fortran order
            frames[:, :, ii] = raw.reshape((line_length, line_count), order="F")

    return frames, head


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        sys.exit("usage: read_ocx_frame.py FILE FRAME [FRAME ...]  (frames are 0-based)")
    data, header = read_ocx_frame(sys.argv[1], [int(a) for a in sys.argv[2:]])
    print(f"Read frames with shape {data.shape}, dtype {data.dtype}")
    print({k: v for k, v in header.items() if k != "config"})
