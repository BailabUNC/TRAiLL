import argparse
import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

def show_trend(df, start_idx, end_idx, frame_interval=10,
               save_format=None, file_prefix=None):
    """
    Plot the trend of:
    - The most significant channel of the selected frames.
    - The average of all channels of the selected frames.
    - The standard deviation of all channels of the selected frames.
    Args:
        df (pd.DataFrame): DataFrame containing channel columns.
        start_idx (int): Start index for the series of frames.
        end_idx (int): End index for the series of frames.
        frame_interval (int): Interval between frames to show in the trend plot.
        save_format (str or None): File format to save the figure ('png', 'svg', or None).
        file_prefix (str): Prefix for the output file name.
    """
    channels = [col for col in df.columns if col.startswith('ch')]
    
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > len(df):
        end_idx = len(df)
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx.")

    # Select frames with the specified interval between start_idx and end_idx
    indices = np.arange(start_idx, end_idx, frame_interval, dtype=int)
    if indices[-1] != end_idx - 1:
        indices = np.append(indices, end_idx - 1)
    df_selected = df.iloc[indices].copy()
    df_selected['frame_idx'] = 10 * (indices - start_idx)  # Convert to milliseconds assuming 100 Hz sampling rate
    df_selected.set_index('frame_idx', inplace=True)
    df_selected = df_selected[channels]
    df_selected = df_selected.reset_index()
    df_selected['most_significant'] = df_selected[channels].mean(axis=0).idxmax()
    df_selected['average'] = df_selected[channels].mean(axis=1)
    df_selected['std_dev'] = df_selected[channels].std(axis=1)

    # Plot the most significant channel, average, and std deviation
    fig, ax = plt.subplots(figsize=(1.5, 1))
    most_significant_channel = df_selected['most_significant'].iloc[0]
    ax.plot(df_selected['frame_idx'], df_selected[most_significant_channel],
            color='#8c64b4', marker='o', lw=0.5, markersize=2.5)
    ax.plot(df_selected['frame_idx'], df_selected['average'],
            color='#666666', lw=0.5, label='Average')
    ax.fill_between(df_selected['frame_idx'],
                    df_selected['average'] - df_selected['std_dev'],
                    df_selected['average'] + df_selected['std_dev'],
                    color='#666666', alpha=0.3)
    ax.set_xlim(df_selected['frame_idx'].min(), df_selected['frame_idx'].max())
    ax.set_ylim(-100, 100)

    # Remove y-axis
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', direction='inout')

    # Place x-axis at y==0
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)

    # Show first, center (no label), and last x-tick labels
    x_start = df_selected['frame_idx'].iloc[0]
    x_end = df_selected['frame_idx'].iloc[-1]
    x_center = (x_start + x_end) // 2
    xticks = [x_start, x_center, x_end]
    xticklabels = [f'{int(x_start)}', '', f'{int(x_end)}']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # ax.set_xlabel('Timestamp (ms)')
    # ax.set_ylabel('Photovoltage (a.u.)')
    plt.tight_layout()

    if save_format in ('png', 'svg'):
        # Compose a reproducible filename with all important argument values
        fname = (
            f"{file_prefix}_start{start_idx}_end{end_idx}_interval{frame_interval}_frames{len(df_selected)}"
        )
        # Add filter info if present in the DataFrame columns
        if any(col.endswith('_filt') for col in df.columns):
            fname += "_filtered"
        fname = fname.replace('None', 'auto').replace('.', 'p')
        save_dir = 'figures/fig2/finger flex trend'
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
        fname += f'.{save_format}'
        plt.savefig(fname, format=save_format, bbox_inches='tight', dpi=600, transparent=True)
        print(f"Saved {save_format.upper()}: {fname}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show a heatmap for a specific frame or a series of frames in a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('-si', '--start-idx', type=int, default=None, help='Start frame index (inclusive) for series.')
    parser.add_argument('-ei', '--end-idx', type=int, default=None, help='End frame index (exclusive) for series.')
    parser.add_argument('--frame-interval', type=int, default=10, help='Interval between frames to show in series view.')
    parser.add_argument('--filter', action='store_true', help='Enable sos bandpass filtering of channels.')
    parser.add_argument('--save-plot', choices=['png', 'svg'], help='Save the series heatmap as either a PNG or SVG file.')
    parser.add_argument('--file-prefix', type=str, default=None, help='Prefix for the saved file name.')
    args = parser.parse_args()

    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['font.size'] = 8

    path = args.file_path

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

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None or args.end_idx > len(df_processed):
        args.end_idx = len(df_processed)
    
    show_trend(df_processed, args.start_idx, args.end_idx, 
               frame_interval=args.frame_interval, save_format=args.save_plot, file_prefix=args.file_prefix)