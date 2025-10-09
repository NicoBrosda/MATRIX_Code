import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *
import cv2


# Get the correct mapping for the matrix array
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

# Define the Analyzer instance and assign dark current / normalization
readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']
A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
A.set_dark_measurement(dark_path, matrix_dark)
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, ['2DLarge_YScan_'], normalization_module=norm_func)

# The movie crit and folder path
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
crit = '2DLarge_movieScan_'
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
upscale = 8
detector_status = True
proportional_breaks = True

# The measurement files and some redundant statistics to allow easier frame settings
folder = folder_path
files = sorted(glob.glob(str(folder / f"*{crit}*.csv")))
n_files = len(files)
if n_files == 0:
    raise FileNotFoundError(f"No files found in {folder}")
frames_per_file, _ = get_frames_per_file(folder, f"*{crit}*.csv")
timefunc = os.path.getmtime
times = np.array([timefunc(f) for f in files], dtype=float)
order = np.argsort(times)
times = times[order]
files = [files[i] for i in order if (crit in files[i] and not '.png' in files[i])]

# The folder with the measurement files to create the movie from + frame selection
zero_frame = 1305
frame_start = 1
frame_space = 1
frames_n = len(files) - 1
frames = [frame_start + i*frame_space for i in range(frames_n)]

results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_111024/MatrixArray/movie_subres{frame_start}_{frame_space}_{frames_n}/')

# Target fps
target_fps = 60
target_length_scale = 1/10
stats = compute_fps_from_files(folder_path, pattern=f"*{crit}*.csv", frames=frames, target_fps=target_fps,
                               length_scale=target_length_scale, verbose=True)
fps_final = stats.get("fps_final", target_fps)

output_mp4 = Path(f'x{target_length_scale:.3f}_{stats['fps_final']:.2f}fps_{frame_start}_{frame_space}_{frames_n}.mp4')
print(output_mp4)

frame_bunch_size = frames_per_file // stats["bunch_size"]

# To generate the intensity limits we make an estimation by going through each file without subframes
readout = lambda x, y: ams_2D_assignment_frame(x, y, channel_assignment=translated_mapping, keep_first_row=True,
                                               frame_bunch_size=1)
A.readout = readout
A.scale = 'nano'
cache = np.array([])
for i in tqdm(range(len(files))):
    if i+1 in frames:
        if len(cache) == 0:
            cache = A.readout(folder_path / files[i], A)
        else:
            cache = np.append(cache, A.readout(folder_path / files[i], A), axis=0)
cache = np.reshape(cache, (len(cache), 11, 11))
cache -= A.dark
cache *= A.norm_factor
cache = A.signal_conversion(cache)
intensity_limits = np.array([0, np.max(cache)*0.8])
# cache = np.transpose(cache, (0, 2, 1))

FG = FrameGenerator(A, inverse=[False, False])
FG.set_dead_pixel_mask(cache[0])

print(intensity_limits)

# The readout for the movie files
readout = lambda x, y: ams_2D_assignment_frame(x, y, channel_assignment=translated_mapping, keep_first_row=True,
                                               frame_bunch_size=frame_bunch_size)
A.readout = readout

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
        x = w - size * 2 - bar_width / 2
        top_left1 = (x - spacing, y - size)
        bottom_right1 = (x - spacing + bar_width, y + size)
        top_left2 = (x + spacing, y - size)
        bottom_right2 = (x + spacing + bar_width, y + size)
        cv2.rectangle(frame_bgr, top_left1, bottom_right1, (0, 0, 255), -1)
        cv2.rectangle(frame_bgr, top_left2, bottom_right2, (0, 0, 255), -1)

    return frame_bgr

# Prepare VideoWriter
h0, w0 = FG.Ny, FG.Nx
out_h, out_w = int(h0*upscale), int(w0*upscale)
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
            cache = A.readout(folder_path / files[i], A)
            tres = ams_2D_assignment_readout(folder_path / files[i], A, channel_assignment=translated_mapping)
        else:
            cache = np.append(cache, A.readout(folder_path / files[i], A), axis=0)
    else:
        continue

    cache = np.reshape(cache, (len(cache), 11, 11))
    cache -= A.dark
    cache *= A.norm_factor
    cache = A.signal_conversion(cache)
    # cache = np.transpose(cache, (0, 2, 1))

    for j in range(len(cache)):
        frame = FG.generate_frame(cache[j])
        frame = frame_to_bgr_uint8(frame, intensity_limits[0], intensity_limits[1], cmap=cmap, upscale=upscale)
        if detector_status:
            writer.write(overlay_detector_status(frame, True))
        else:
            writer.write(frame)
        total_video_frames += 1

    if detector_status:
        frame = FG.generate_frame(cache[-1])
        frame = frame_to_bgr_uint8(frame, intensity_limits[0], intensity_limits[1], cmap=cmap, upscale=upscale)
        frame = overlay_detector_status(frame, False)

    # Add "pause" between files based on dt
    if k < len(dt_list):
        if proportional_breaks:
            n_pause_frames = int(round(dt_list[k]*target_length_scale / dt_ref))
        else:
            n_pause_frames = int(1 / dt_ref)
        if n_pause_frames > 0:
            for _ in range(n_pause_frames):
                writer.write(frame)  # repeat last frame
            total_video_frames += n_pause_frames
    k += 1

writer.release()
print(f"Video written to {output_mp4}, total frames: {total_video_frames}")

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

