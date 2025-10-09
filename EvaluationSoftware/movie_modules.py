import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import distance_transform_edt
import cv2
from tqdm import tqdm
import matplotlib
from matplotlib import cm
from Plot_Methods.plot_standards import *
from EvaluationSoftware.main import *


def get_frames_per_file(folder, criterion):
    # Find first matching file
    files = sorted(glob.glob(os.path.join(folder, f"*{criterion}*")))
    if not files:
        raise FileNotFoundError(f"No files found with criterion '{criterion}' in {folder}")

    first_file = files[0]

    # Count rows (fast: count lines instead of full read)
    with open(first_file, "r") as f:
        n_lines = sum(1 for _ in f)

    # Frames per file = (rows - header) + extracted-header-row
    frames_per_file = (n_lines - 1) + 1

    return frames_per_file, first_file


def compute_fps_from_files(folder, pattern="*.csv",
                           frames_per_file=None,
                           use_ctime=False,
                           frames=None,
                           verbose=True):
    """
    Estimate FPS from file timestamps, with optional frame selection and movie stats.

    Parameters
    ----------
    folder : str or Path
        Folder containing the files.
    pattern : str
        Glob pattern to select files (default '*.csv').
    frames_per_file : int
        How many logical frames are in each file (default 1).
    use_ctime : bool
        Use creation time (getctime) instead of modification time (getmtime).
    frames : list or tuple, optional
        [start, stop, n_frames] to select specific frames.
    target_movie_length_sec : float, optional
        Desired length of the final movie.
    verbose : bool
        Print summary stats.

    Returns
    -------
    dict
        Dictionary containing all timing statistics and file list.
    """
    folder = Path(folder)
    files = sorted(glob.glob(str(folder / pattern)))
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No files found in {folder} matching '{pattern}'")

    if frames_per_file is None:
        frames_per_file, _ = get_frames_per_file(folder, pattern)

    # pick timestamp function
    timefunc = os.path.getctime if use_ctime else os.path.getmtime
    times = np.array([timefunc(f) for f in files], dtype=float)
    order = np.argsort(times)
    times = times[order]
    files = [files[i] for i in order]

    # Handle frame selection
    if frames is None:
        frames_idx = np.arange(len(files))
    elif len(frames) > 3:
        frames_idx = np.array(frames)
    else:
        start, stop, n_sel = frames
        frames_idx = np.linspace(start, stop, n_sel, dtype=int)

    files = np.array(files)[frames_idx]
    n_files = len(files)
    times = times[frames_idx]

    # Intervals between consecutive files
    dts_files = np.diff(times) if len(times) > 1 else np.array([], dtype=float)
    duration_files = times[-1] - times[0] if len(times) > 1 else 0.0

    total_frames = frames_per_file * n_files
    fps_est = (total_frames - 1) / duration_files if duration_files > 1e-9 and total_frames > 1 else None
    fps_orig = (n_files - 1) / duration_files if duration_files > 1e-9 else None

    # Intervals between consecutive files
    if len(times) > 1:
        dts = np.diff(times)
        duration = times[-1] - times[0]
    else:
        dts = np.array([], dtype=float)
        duration = 0.0

    stats = {
        "n_files": n_files,
        "frames_per_file": frames_per_file,
        "total_frames": total_frames,
        "duration_s": float(duration_files),
        "fps_est": float(fps_est) if fps_est is not None else None,
        "fps_orig": float(fps_orig) if fps_orig is not None else None,
        "mean_dt": float(np.mean(dts)) if dts.size else None,
        "median_dt": float(np.median(dts)) if dts.size else None,
        "std_dt": float(np.std(dts)) if dts.size else None,
        "min_dt": float(np.min(dts)) if dts.size else None,
        "max_dt": float(np.max(dts)) if dts.size else None,
        "frames_idx": frames_idx,
        "files": files,
        "times": times
    }

    if verbose:
        print(f"Found {n_files} files (frames_per_file={frames_per_file})")
        print("First file:", files[0], "time:", datetime.fromtimestamp(times[0]))
        print("Last  file:", files[-1], "time:", datetime.fromtimestamp(times[-1]))
        print(f"Duration (s): {duration:.6f} | Duration (min): {duration / 60:.6f}")
        if fps_est is not None:
            print(f"Estimated fps = (total_frames-1)/duration = {fps_est:.3f} fps")
            print(f"Original fps = (measurement_files-1)/duration = {fps_orig:.3f} fps")
        else:
            print("Not enough time span to estimate fps.")
        if dts.size:
            print("Inter-file dt stats (s): mean %.6f, median %.6f, std %.6f, min %.6f, max %.6f" %
                  (stats["mean_dt"], stats["median_dt"], stats["std_dt"], stats["min_dt"], stats["max_dt"]))
        else:
            print("Only one file — no inter-file intervals available.")

    return stats


def compute_fps_from_files(folder, pattern="*.csv",
                           frames_per_file=None,
                           use_ctime=False,
                           frames=None,
                           target_fps=None,
                           length_scale=1.0,
                           verbose=True):
    """
    Estimate FPS from file timestamps, with optional frame selection,
    movie stats, and fps scaling.

    Parameters
    ----------
    folder : str or Path
        Folder containing the files.
    pattern : str
        Glob pattern to select files (default '*.csv').
    frames_per_file : int
        How many logical frames are in each file (default 1).
    use_ctime : bool
        Use creation time (getctime) instead of modification time (getmtime).
    frames : list or tuple, optional
        [start, stop, n_frames] to select specific frames.
    target_fps : float, optional
        Desired fps of the output movie (default None → keep scaled fps).
    length_scale : float, optional
        Scaling factor for movie duration (1.0 = same, 0.1 = 10× faster, 2.0 = 2× slower).
    verbose : bool
        Print summary stats.

    Returns
    -------
    dict
        Dictionary containing all timing statistics and file list.
    """
    folder = Path(folder)
    files = sorted(glob.glob(str(folder / pattern)))
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No files found in {folder} matching '{pattern}'")

    if frames_per_file is None:
        frames_per_file, _ = get_frames_per_file(folder, pattern)

    # pick timestamp function
    timefunc = os.path.getctime if use_ctime else os.path.getmtime
    times = np.array([timefunc(f) for f in files], dtype=float)
    order = np.argsort(times)
    times = times[order]
    files = [files[i] for i in order]

    # Handle frame selection
    if frames is None:
        frames_idx = np.arange(len(files))
    else:
        frames_idx = np.array(frames)

    files = np.array(files)[frames_idx]
    n_files = len(files)
    times = times[frames_idx]

    # Intervals between consecutive files
    dts = np.diff(times) if len(times) > 1 else np.array([], dtype=float)
    duration_files = times[-1] - times[0] if len(times) > 1 else 0.0

    total_frames = frames_per_file * n_files
    fps_est = (total_frames - 1) / duration_files if duration_files > 1e-9 and total_frames > 1 else None
    fps_orig = (n_files - 1) / duration_files if duration_files > 1e-9 else None

    # Effective fps after scaling
    length_scale = 1 if length_scale is None else length_scale
    fps_eff = fps_est / length_scale if fps_est else None

    # Compute bunch size if target_fps is given
    bunch_size = None
    if fps_eff and target_fps:
        bunch_size = max(1, int(round(fps_eff / target_fps)))
        fps_final = fps_eff / bunch_size
    else:
        fps_final = fps_eff

    stats = {
        "n_files": n_files,
        "frames_per_file": frames_per_file,
        "total_frames": total_frames,
        "duration_s": float(duration_files),
        "fps_est": float(fps_est) if fps_est is not None else None,
        "fps_orig": float(fps_orig) if fps_orig is not None else None,
        "fps_eff": float(fps_eff) if fps_eff is not None else None,
        "fps_final": float(fps_final) if fps_final is not None else None,
        "length_scale": length_scale,
        "target_fps": target_fps,
        "bunch_size": bunch_size,
        "frames_idx": frames_idx,
        "files": files,
        "times": times,
        "mean_dt": float(np.mean(dts)) if dts.size else None,
        "median_dt": float(np.median(dts)) if dts.size else None,
        "std_dt": float(np.std(dts)) if dts.size else None,
        "dt": dts
    }

    if verbose:
        print(f"Found {n_files} files (frames_per_file={frames_per_file})")
        print("First file:", files[0], "time:", datetime.fromtimestamp(times[0]))
        print("Last  file:", files[-1], "time:", datetime.fromtimestamp(times[-1]))
        print(f"Duration (s): {duration_files:.6f} | Duration (min): {duration_files / 60:.6f}")
        if fps_est is not None:
            print(f"Estimated fps from data = {fps_est:.3f}")
            print(f"Original fps (files only) = {fps_orig:.3f}")
            print(f"Effective fps after length_scale={length_scale} → {fps_eff:.3f}")
            print(f"Effective length after length_scale (s): {duration_files*length_scale:.6f}  | Duration (min): {duration_files*length_scale / 60:.6f}")

        if target_fps:
            print(f"Target fps = {target_fps}, bunch_size = {bunch_size}, final fps = {fps_final:.3f}")

        if dts.size:
            print("Inter-file dt stats (s): mean %.6f, median %.6f, std %.6f" %
                  (stats["mean_dt"], stats["median_dt"], stats["std_dt"]))
        else:
            print("Only one file — no inter-file intervals available.")

    return stats


class FrameGenerator2:
    """
    Generate 2D frames from diode signals with proper spacing and optional interpolation.
    The geometry and indices are precomputed for efficiency.
    """

    def __init__(self, instance, inverse=[False, False]):
        import numpy as np

        self.inverse = inverse
        self.Nx_diode, self.Ny_diode = instance.diode_dimension
        self.dx, self.dy = instance.diode_size
        self.sx, self.sy = instance.diode_spacing

        # Compute number of signal pixels per diode
        self.px_per_diode_x = max(1, int(np.round(self.dx / self.sx)))
        self.px_per_diode_y = max(1, int(np.round(self.dy / self.sy)))

        # Compute gaps in pixels
        self.gap_x = max(0, int(np.round(self.sx / self.sx)))  # mindestens 0
        self.gap_y = max(0, int(np.round(self.sy / self.sy)))

        # Total frame size
        self.Nx = self.Nx_diode * self.px_per_diode_x + (self.Nx_diode - 1) * self.gap_x
        self.Ny = self.Ny_diode * self.px_per_diode_y + (self.Ny_diode - 1) * self.gap_y

        # Precompute start indices of each diode in the frame
        self.x_starts = np.cumsum([0] + [self.px_per_diode_x + self.gap_x] * (self.Nx_diode - 1))
        self.y_starts = np.cumsum([0] + [self.px_per_diode_y + self.gap_y] * (self.Ny_diode - 1))

    def generate_frame(self, signals):
        """
        Generate one frame from 2D signal array using precomputed geometry.

        Parameters
        ----------
        signals : 2D np.array of shape (Ny_diode, Nx_diode)
            Signal values for each diode.

        Returns
        -------
        frame : 2D np.array of shape (Ny, Nx)
        """
        import numpy as np

        frame = np.zeros((self.Ny, self.Nx), dtype=float)

        # Fill diode pixels
        for i in range(self.Nx_diode):
            for j in range(self.Ny_diode):
                x0 = self.x_starts[i]
                y0 = self.y_starts[j]
                frame[y0:y0 + self.px_per_diode_y, x0:x0 + self.px_per_diode_x] = signals[j, i]

        # Interpolate gaps in X
        if self.gap_x > 0:
            for i in range(self.Nx_diode - 1):
                x_start = self.x_starts[i] + self.px_per_diode_x
                x_end = self.x_starts[i + 1]
                frame[:, x_start:x_end] = (frame[:, self.x_starts[i] + self.px_per_diode_x - 1][:, None] +
                                           frame[:, self.x_starts[i + 1]][:, None]) / 2

        # Interpolate gaps in Y
        if self.gap_y > 0:
            for j in range(self.Ny_diode - 1):
                y_start = self.y_starts[j] + self.px_per_diode_y
                y_end = self.y_starts[j + 1]
                frame[y_start:y_end, :] = (frame[self.y_starts[j] + self.px_per_diode_y - 1, :][None, :] +
                                           frame[self.y_starts[j + 1], :][None, :]) / 2

        # Apply inversion
        if self.inverse[0]:
            frame = frame[::-1, :]
        if self.inverse[1]:
            frame = frame[:, ::-1]

        return frame


import numpy as np
from scipy.ndimage import convolve


class FrameGenerator:
    def __init__(self, instance, inverse=[False, False]):
        import numpy as np

        self.inverse = inverse
        self.Nx_diode, self.Ny_diode = instance.diode_dimension
        self.dx, self.dy = instance.diode_size
        self.sx, self.sy = instance.diode_spacing

        self.xextent, self.yextent = (self.dx+self.sx)*self.Nx_diode, (self.dy+self.sy)*self.Ny_diode
        # Compute number of signal pixels per diode
        self.px_per_diode_x = max(1, int(np.round(self.dx / self.sx)))
        self.px_per_diode_y = max(1, int(np.round(self.dy / self.sy)))

        # Compute gaps in pixels
        self.gap_x = max(0, int(np.round(self.sx / self.sx)))  # mindestens 0
        self.gap_y = max(0, int(np.round(self.sy / self.sy)))

        # Total frame size
        self.Nx = self.Nx_diode * self.px_per_diode_x + (self.Nx_diode - 1) * self.gap_x
        self.Ny = self.Ny_diode * self.px_per_diode_y + (self.Ny_diode - 1) * self.gap_y

        # Precompute start indices of each diode in the frame
        self.x_starts = np.cumsum([0] + [self.px_per_diode_x + self.gap_x] * (self.Nx_diode - 1))
        self.y_starts = np.cumsum([0] + [self.px_per_diode_y + self.gap_y] * (self.Ny_diode - 1))

        # Dead-Pixel-Maske (optional, später setzbar)
        self.dead_pixel_mask = None

        # Kernel für 4-Nachbarschaft
        self.kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=float)

    def set_dead_pixel_mask(self, frame):
        """Define dead pixel mask once from a reference frame (0 values)."""
        self.dead_pixel_mask = (frame == 0)

    def interpolate_dead_pixels(self, frame):
        """Replace dead pixels with average of non-dead neighbors (vectorized)."""
        if self.dead_pixel_mask is None:
            return frame

        # Sum of neighbors
        neighbor_sum = convolve(frame, self.kernel, mode="constant", cval=0.0)
        # Count of neighbors (but only nonzero contributions)
        neighbor_count = convolve((frame != 0).astype(float), self.kernel, mode="constant", cval=0.0)

        # Avoid div by zero
        interp_values = np.zeros_like(frame, dtype=float)
        valid = neighbor_count > 0
        interp_values[valid] = neighbor_sum[valid] / neighbor_count[valid]

        # Replace dead pixels
        frame_fixed = frame.copy()
        frame_fixed[self.dead_pixel_mask] = interp_values[self.dead_pixel_mask]

        return frame_fixed

    def interpolate_dead_pixels2(self, frame, threshold=1e-9):
        """
        Replace isolated dead pixels (or small clusters) with the average
        of their valid 4-neighbors. Much faster and more edge-preserving
        for small matrices with few defects.
        """
        if self.dead_pixel_mask is None:
            return frame

        frame_fixed = frame.copy()
        Ny, Nx = frame.shape
        dead_y, dead_x = np.where(self.dead_pixel_mask)

        for y, x in zip(dead_y, dead_x):
            neighbors = []

            # Check 4-neighborhood (up, down, left, right)
            if y > 0 and frame[y - 1, x] > threshold:
                neighbors.append(frame[y - 1, x])
            if y < Ny - 1 and frame[y + 1, x] > threshold:
                neighbors.append(frame[y + 1, x])
            if x > 0 and frame[y, x - 1] > threshold:
                neighbors.append(frame[y, x - 1])
            if x < Nx - 1 and frame[y, x + 1] > threshold:
                neighbors.append(frame[y, x + 1])

            if neighbors:
                # Mean of valid neighbors
                frame_fixed[y, x] = np.mean(neighbors)
            else:
                # Fallback: use global mean of nonzero pixels
                nonzero = frame[frame > threshold]
                frame_fixed[y, x] = np.mean(nonzero) if nonzero.size > 0 else 0.0

        return frame_fixed

    def generate_frame(self, signals):
        """
        Generate one frame from 2D signal array using precomputed geometry.
        Includes gap interpolation and dead pixel replacement.
        """
        # Replace dead pixels (vectorized)
        signals = self.interpolate_dead_pixels(signals)

        frame = np.zeros((self.Ny, self.Nx), dtype=float)

        # Fill diode pixels
        for i in range(self.Nx_diode):
            for j in range(self.Ny_diode):
                x0 = self.x_starts[i]
                y0 = self.y_starts[j]
                frame[y0:y0 + self.px_per_diode_y,
                      x0:x0 + self.px_per_diode_x] = signals[j, i]

        # Interpolate gaps in X
        if self.gap_x > 0:
            for i in range(self.Nx_diode - 1):
                x_start = self.x_starts[i] + self.px_per_diode_x
                x_end = self.x_starts[i + 1]
                frame[:, x_start:x_end] = (
                    frame[:, self.x_starts[i] + self.px_per_diode_x - 1][:, None]
                    + frame[:, self.x_starts[i + 1]][:, None]
                ) / 2

        # Interpolate gaps in Y
        if self.gap_y > 0:
            for j in range(self.Ny_diode - 1):
                y_start = self.y_starts[j] + self.px_per_diode_y
                y_end = self.y_starts[j + 1]
                frame[y_start:y_end, :] = (
                    frame[self.y_starts[j] + self.px_per_diode_y - 1, :][None, :]
                    + frame[self.y_starts[j + 1], :][None, :]
                ) / 2


        # Apply inversion
        if self.inverse[0]:
            frame = frame[::-1, :]
        if self.inverse[1]:
            frame = frame[:, ::-1]

        return frame


def generate_movie(analyzer, folder_path, crit, output_path, output_name=None, frame_select=None, zero_frame=None,
                   target_fps=60, length_scale=1., movie_res=4, detector_status=True, breaks: float | str = 'proportional',
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"]),
                   compress=True, return_info=False):    #
    # The measurement files and some redundant statistics to allow easier frame settings
    files = sorted(glob.glob(str(folder_path / f"*{crit}*.csv")))
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No files found in {folder_path}")
    frames_per_file, _ = get_frames_per_file(folder_path, f"*{crit}*.csv")
    timefunc = os.path.getmtime
    times = np.array([timefunc(f) for f in files], dtype=float)
    order = np.argsort(times)
    times = times[order]
    files = [files[i] for i in order if (crit in files[i] and not '.png' in files[i])]

    # The folder with the measurement files to create the movie from + frame selection
    if frame_select:
        frame_start = frame_select[0]
        frame_space = frame_select[1]
        frames_n = frame_select[2]
    else:
        frame_start = 1
        frame_space = 1
        frames_n = len(files) - 1
    frames = [frame_start + i*frame_space for i in range(frames_n)]

    results_path = output_path / Path(f'movie_subres{frame_start}_{frame_space}_{frames_n}/')

    # Target fps
    stats = compute_fps_from_files(folder_path, pattern=f"*{crit}*.csv", frames=frames, target_fps=target_fps,
                                   length_scale=length_scale, verbose=True)

    if return_info:
        return stats

    fps_final = stats.get("fps_final", target_fps)

    if output_name:
        output_mp4 = str(output_name)
    else:
        output_mp4 = Path(f'x{length_scale:.3f}_{stats['fps_final']:.2f}fps_{frame_start}_{frame_space}_{frames_n}.mp4')

    if zero_frame:
        output_mp4 = Path(f'_zero{zero_frame}_' + str(output_mp4))

    frame_bunch_size = frames_per_file // stats["bunch_size"]
    # To generate the intensity limits we make an estimation by going through each file without subframes
    # readout = lambda x, y: ams_2D_assignment_frame(x, y, channel_assignment=translated_mapping, keep_first_row=True, frame_bunch_size=1)
    # analyzer.readout = readout
    # analyzer.scale = 'nano'

    cache = np.array([])
    for i in tqdm(range(len(files))):
        if i+1 in frames:
            if len(cache) == 0:
                cache = analyzer.readout(folder_path / files[i], analyzer)
            else:
                cache = np.append(cache, analyzer.readout(folder_path / files[i], analyzer), axis=0)
        if zero_frame and i+1 == zero_frame:
            zero = analyzer.readout(folder_path / files[i], analyzer)
            zero = np.reshape(zero, (11, 11))

    cache = np.reshape(cache, (len(cache), 11, 11))
    cache -= analyzer.dark
    cache *= analyzer.norm_factor
    cache = analyzer.signal_conversion(cache)
    if zero_frame:
        zero -= analyzer.dark
        zero *= analyzer.norm_factor
        zero = analyzer.signal_conversion(zero)
        cache = np.abs(cache - zero)
    intensity_limits = np.array([0, np.max(cache)*0.8])
    # cache = np.transpose(cache, (0, 2, 1))

    FG = FrameGenerator(analyzer, inverse=[False, False])
    FG.set_dead_pixel_mask(cache[0])

    # Save a single frame into a maptlotib plot as reference
    poster = True
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    # Add colorbar using dummy ScalarMappable
    ax.set_xlim(0, FG.xextent)
    ax.set_ylim(0, FG.yextent)
    norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if intensity_limits[0] > 0:
        bar = fig.colorbar(sm, ax=ax, extend='both')
    else:
        bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel(r'Position x (mm)')
    ax.set_ylabel(r'Position y (mm)')
    bar.set_label(f'Signal Current ({scale_dict[analyzer.scale][1]}A)')
    format_save(save_path=results_path, save_name=str(output_mp4)[:-4], dpi=300, save_format=save_format,
                fig=fig, transparent=True)

    print(intensity_limits)

    # The readout for the movie files
    readout = analyzer.readout
    analyzer.readout = lambda x, y: readout(x, y, frame_bunch_size=frame_bunch_size)

    # ---------- PASS 2: stream to video ----------
    os.makedirs(results_path, exist_ok=True)

    print("Pass 2: streaming frames to MP4...")

    # colormap: we need a mapping function that maps float frame -> uint8 RGB
    # If user provided a matplotlib colormap object `cmap`, use it.
    if not isinstance(cmap, matplotlib.colors.Colormap):
        cmap = cm.get_cmap("viridis")

    def frame_to_bgr_uint8(frame2d, vmin, vmax, cmap=cmap, upscale=4):
        # transpose as requested to match alignment
        frame2d = frame2d.T  # fast view transpose for contiguousness may need copy
        # normalize to 0..1
        if vmax > vmin:
            norm = (frame2d - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(frame2d)
        norm = np.clip(norm, 0.0, 1.0)

        # apply colormap (returns RGBA floats 0..1)
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        # resize if upscale > 1
        if upscale != 1:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (w*upscale, h*upscale), interpolation=cv2.INTER_NEAREST)

        # convert RGB to BGR for OpenCV writer
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # Detector status
    def overlay_detector_status(frame_bgr, active: bool):
        """
        Overlay detector status on the video frame.

        - Red circle = recording (active)
        - Pause symbol "II" = inactive
        """
        h, w = frame_bgr.shape[:2]

        # Indicator position and size
        size = max(7, w // 20)
        x, y = w - size * 2, size * 2

        if active:
            # Draw red circle for recording
            cv2.circle(frame_bgr, (x, y), size, (0, 0, 255), -1)  # BGR
        else:
            # Draw "pause" symbol: two vertical white rectangles
            bar_width = size // 3
            spacing = bar_width
            x = int(round(w - size * 2 - bar_width / 2))
            top_left1 = (x - spacing, y - size)
            bottom_right1 = (x - spacing + bar_width, y + size)
            top_left2 = (x + spacing, y - size)
            bottom_right2 = (x + spacing + bar_width, y + size)
            cv2.rectangle(frame_bgr, top_left1, bottom_right1, (0, 0, 255), -1)
            cv2.rectangle(frame_bgr, top_left2, bottom_right2, (0, 0, 255), -1)

        return frame_bgr

    # Prepare VideoWriter
    h0, w0 = FG.Ny, FG.Nx
    out_h, out_w = int(h0*movie_res), int(w0*movie_res)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(results_path / str(output_mp4), fourcc, fps_final, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open.")

    total_video_frames = 0

    # Calculate timing breaks
    dt_list = list(stats.get("dt", []))
    if len(dt_list) < len(frames) - 1:
        mean_dt = stats.get("mean_dt", None)
        dt_list += [mean_dt] * (len(files) - 1 - len(dt_list)) if mean_dt else [0] * (len(files) - 1 - len(dt_list))
    dt_ref = 1.0 / fps_final

    # Stream frames
    k = 0
    for i in tqdm(range(len(files))):
        cache = np.array([])

        if i+1 in frames:
            if len(cache) == 0:
                cache = analyzer.readout(folder_path / files[i], analyzer)
            else:
                cache = np.append(cache, analyzer.readout(folder_path / files[i], analyzer), axis=0)
        else:
            continue

        cache = np.reshape(cache, (len(cache), 11, 11))
        cache -= analyzer.dark
        cache *= analyzer.norm_factor
        cache = analyzer.signal_conversion(cache)
        if zero_frame:
            cache = np.abs(cache - zero)

        for j in range(len(cache)):
            frame = FG.generate_frame(cache[j])
            frame = frame_to_bgr_uint8(frame, intensity_limits[0], intensity_limits[1], cmap=cmap, upscale=movie_res)
            if detector_status:
                writer.write(overlay_detector_status(frame, True))
            else:
                writer.write(frame)
            total_video_frames += 1

        # Add "pause" between files based on dt
        if k < len(dt_list):
            if breaks == 'proportional':
                n_pause_frames = int(round(dt_list[k]*length_scale / dt_ref))
            elif breaks:
                n_pause_frames = int(breaks / dt_ref)
            else:
                continue

            if detector_status:
                frame = FG.generate_frame(cache[-1])
                frame = frame_to_bgr_uint8(frame, intensity_limits[0], intensity_limits[1], cmap=cmap,
                                           upscale=movie_res)
                frame = overlay_detector_status(frame, False)

            if n_pause_frames > 0:
                for _ in range(n_pause_frames):
                    writer.write(frame)  # repeat last frame
                total_video_frames += n_pause_frames
        k += 1

    writer.release()
    print(f"Video written to {output_mp4}, total frames: {total_video_frames}")

    if compress:
        # Step 2: Compress it
        import subprocess
        compressed_mp4 = results_path / output_mp4.with_name(output_mp4.stem + "compressed.mp4")
        output_path = results_path / output_mp4
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-vcodec", "libx264",
            "-crf", "30",
            "-preset", "slow",
            "-movflags", "+faststart",
            str(compressed_mp4)
        ])

        # Delete original if compression succeeded
        if compressed_mp4.exists() and compressed_mp4.stat().st_size > 0:
            output_path.unlink()  # removes original file
            output_path = compressed_mp4
            print(f"Deleted uncompressed file, using {output_path.name} instead.")
        else:
            print("Warning: compressed file missing or empty; keeping original.")

    return stats