import argparse
import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from cmap import Colormap
from scipy.ndimage import zoom

def show_frame_heatmap(df, frame_idx, channel_prefix='ch', vmin=None, vmax=None, save_png=False, png_prefix='frame_heatmap'):
    """
    Show a heatmap of all channel values for a specific frame (row) in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing channel columns.
        frame_idx (int): Index of the frame (row) to visualize.
        channel_prefix (str): Prefix for channel columns (default: 'ch').
        vmin, vmax: Color scale limits for the heatmap.
        save_png (bool): Whether to save the figure as a PNG file.
        png_prefix (str): Prefix for the PNG file name.
    """
    channels = [col for col in df.columns if col.startswith(channel_prefix)]
    values = df.loc[frame_idx, channels].values
    print(values)

    heatmap = values.reshape(6, 8)
    # heatmap = np.fliplr(heatmap)
    heatmap = zoom(heatmap, (5, 5), order=3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)  # Use a single subplot for simplicity
    im = ax.imshow(
        heatmap,
        aspect='equal',
        cmap=Colormap('cmocean:balance').to_mpl(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    if save_png:
        fname = f"{png_prefix}_frame{frame_idx}_vmin{vmin}_vmax{vmax}"
        fname = fname.replace('None', 'auto').replace('.', 'p')
        save_dir = f'figures/fig2/distal radionulnar joint/heatmap frames'
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
        fname += '.png'
        plt.savefig(fname, format='png', bbox_inches='tight', pad_inches=0, dpi=600, transparent=True)
        print(f"Saved PNG: {fname}")

    plt.show()

def show_series_heatmap(df, vmin=None, vmax=None, z_offset_multiplier=None,
                        start_idx=None, end_idx=None, num_frames_to_show=10, 
                        save_png=False, png_prefix='series_heatmap'):
    """
    Show a series of heatmaps stacked in a 3D plot.

    Args:
        df (pd.DataFrame): DataFrame containing channel columns.
        vmin, vmax: Color scale limits for the heatmap.
        z_offset_multiplier (float): Vertical offset between heatmaps in the 3D plot.
        start_idx (int): Start frame index (inclusive).
        end_idx (int): End frame index (exclusive).
        num_frames_to_show (int): Number of frames to display.
        save_png (bool): Whether to save the figure as a PNG file.
        png_prefix (str): Prefix for the PNG file name.
    """
    channels = [col for col in df.columns if col.startswith('ch')]
    total_frames = len(df)

    # Determine the range of frames to use
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > total_frames:
        end_idx = total_frames

    if start_idx >= end_idx or start_idx < 0 or end_idx > total_frames:
        print('Invalid start_idx or end_idx for the DataFrame.')
        return

    available_frames = end_idx - start_idx
    if available_frames < num_frames_to_show:
        interval = 1
        num_frames_to_show = available_frames
        print(f'Warning: Requested {num_frames_to_show} frames, but only {available_frames} are available. Displaying all available frames.')
    else:
        interval = max((available_frames - 1) // (num_frames_to_show - 1), 1) if num_frames_to_show > 1 else 1

    selected_frames_data = []
    for i in range(num_frames_to_show):
        frame_idx = start_idx + i * interval
        if frame_idx < end_idx:
            values = df.iloc[frame_idx][channels].values
            if len(values) == 48:
                heatmap = values.reshape(6, 8)
                zoomed_heatmap = zoom(heatmap, (5, 5), order=3)
                selected_frames_data.append(zoomed_heatmap)
            else:
                print(f'Warning: Frame {frame_idx} does not have 48 channels. Skipping.')
        else:
            break

    if not selected_frames_data:
        print('No valid frames to display in series heatmap.')
        return

    # Prepare grid for surface plot (interpolated size)
    x = np.linspace(0, 7, 8 * 5)
    y = np.linspace(0, 5, 6 * 5)
    X, Y = np.meshgrid(x, y)

    # Figure setup
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1.25, 1, 3))
    ax.view_init(elev=20, azim=-60)
    ax.set_proj_type('ortho')

    # Set grid lines to light gray
    ax.xaxis._axinfo['grid'].update(color='lightgray', linestyle='-', linewidth=0.7)
    ax.yaxis._axinfo['grid'].update(color='lightgray', linestyle='-', linewidth=0.7)
    ax.zaxis._axinfo['grid'].update(color='lightgray', linestyle='-', linewidth=0.7)

    if z_offset_multiplier is None:
        z_offset_multiplier = (vmax - vmin) / 2 if vmax is not None and vmin is not None else 50
    for idx, Z_data in enumerate(selected_frames_data):
        ax.plot_surface(
            X, Y, Z_data + idx * z_offset_multiplier,
            alpha=0.9, cmap=Colormap('cmocean:balance').to_mpl(),
            vmin=vmin + idx * z_offset_multiplier if vmin is not None else None,
            vmax=vmax + idx * z_offset_multiplier if vmax is not None else None,
        )

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_png:
        # Compose a reproducible filename with all important argument values
        fname = (
            f'{png_prefix}_vmin{vmin}_vmax{vmax}_start{start_idx}_end{end_idx}_frames{num_frames_to_show}'
            f'_zoff{z_offset_multiplier}'
        )
        fname = fname.replace('None', 'auto').replace('.', 'p')
        save_dir = f'figures/fig2/distal radionulnar joint/heatmap frames'
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
        fname += '.png'
        plt.savefig(fname, format='png', bbox_inches='tight', pad_inches=0, dpi=600, transparent=True)
        print(f'Saved PNG: {fname}')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show a heatmap for a specific frame or a series of frames in a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--frame-idx', type=int, default=None, help='Frame index to visualize (for single heatmap).')
    parser.add_argument('--vmin', type=float, default=-150, help='Minimum value for color scale.')
    parser.add_argument('--vmax', type=float, default=150, help='Maximum value for color scale.')
    parser.add_argument('--filter', action='store_true', help='Enable sos bandpass filtering of channels.')
    parser.add_argument('--series', action='store_true', help='Show a series of heatmaps in 3D.')
    parser.add_argument('--num-series-frames', type=int, default=10, help='Number of frames to show in series view.')
    parser.add_argument('-si', '--start-idx', type=int, default=None, help='Start frame index (inclusive) for series.')
    parser.add_argument('-ei', '--end-idx', type=int, default=None, help='End frame index (exclusive) for series.')
    parser.add_argument('-zom', '--z-offset-multiplier', type=float, default=None, help='Vertical offset between heatmaps in the 3D plot.')
    parser.add_argument('--save-png', action='store_true', help='Save the series heatmap as a PNG file.')
    parser.add_argument('--png-prefix', type=str, default=None, help='Prefix for the PNG file name.')
    args = parser.parse_args()

    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['font.size'] = 14

    path = args.file_path
    frame_idx = args.frame_idx

    df = pd.read_csv(path, delimiter=',', parse_dates=['timestamp'])
    df['seconds'] = df['timestamp'].apply(lambda x: x.timestamp())
    df['seconds'] -= df['seconds'][0]

    channels = [col for col in df.columns if col.startswith('ch')]
    num_channels = len(channels)

    if args.filter:
        sos = butter(2, [0.2, 30], 'bandpass', output='sos', fs=100)
        for ch in channels:
            df[ch + '_filt'] = sosfiltfilt(sos, df[ch])
        filtered_channels = [ch + '_filt' for ch in channels]
        df_processed = df.copy()
        df_processed = df_processed[filtered_channels]
        df_processed.columns = [col.replace('_filt', '') for col in df_processed.columns]
    else:
        df_processed = df[channels].copy()

    if args.series:
        if args.frame_idx is not None:
            print('Warning: --frame_idx is ignored when --series is specified.')
        png_prefix = args.png_prefix if args.png_prefix is not None else 'series_heatmap'
        show_series_heatmap(
            df_processed,
            vmin=args.vmin,
            vmax=args.vmax,
            num_frames_to_show=args.num_series_frames,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            z_offset_multiplier=args.z_offset_multiplier,
            save_png=args.save_png,
            png_prefix=png_prefix
        )
    elif args.frame_idx is not None:
        png_prefix = args.png_prefix if args.png_prefix is not None else 'frame_heatmap'
        show_frame_heatmap(
            df_processed,
            frame_idx=args.frame_idx,
            vmin=args.vmin,
            vmax=args.vmax,
            save_png=args.save_png,
            png_prefix=png_prefix
        )
    else:
        print('Please specify either --frame-idx for a single heatmap or --series for a series of heatmaps.')