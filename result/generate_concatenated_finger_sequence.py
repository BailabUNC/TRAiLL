import argparse
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

class FingerSequenceProcessor:
    def __init__(self, cutoff=30, fs=100, order=4, num_instances=None, min_instance_length=10):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.num_instances = num_instances
        self.min_instance_length = min_instance_length

    def apply_butterworth(self, data):
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        sos = butter(self.order, normal_cutoff, btype='low', output='sos')
        filtered_data = sosfiltfilt(sos, data, axis=0)
        return filtered_data

    def align_onset(self, active_data, preceding_data, following_data,
                    pre_trigger_points=10, onset_threshold_factor=0.5):
        diff_norm = np.linalg.norm(np.diff(active_data, axis=0), axis=1)
        threshold = onset_threshold_factor * np.max(diff_norm)
        onset_indices = np.where(diff_norm > threshold)[0]
        onset_index_active = int(onset_indices[0]) if onset_indices.size > 0 else 0

        if preceding_data is not None:
            extra_needed = pre_trigger_points - onset_index_active
            if extra_needed > 0:
                num_rows_preceding = preceding_data.shape[0]
                rows_to_prepend = preceding_data[max(0, num_rows_preceding - extra_needed):]
                concatenated_data = np.concatenate((rows_to_prepend, active_data), axis=0)
                trigger_offset = rows_to_prepend.shape[0] + onset_index_active
            else:
                num_rows_to_trim = -extra_needed
                trimmed_data = active_data[num_rows_to_trim:]
                trigger_offset = pre_trigger_points
                if following_data is not None and num_rows_to_trim > 0:
                    rows_to_append = following_data[:num_rows_to_trim]
                    concatenated_data = np.concatenate((trimmed_data, rows_to_append), axis=0)
                else:
                    concatenated_data = trimmed_data
            return concatenated_data, trigger_offset
        else:
            effective_start = max(0, onset_index_active - pre_trigger_points)
            trigger_offset = onset_index_active - effective_start
            trimmed_data = active_data[effective_start:]
            return trimmed_data, trigger_offset

    def trim_tail(self, signal, onset_threshold_factor=0.5, post_trigger_points=10):
        signal_length = signal.shape[0]
        if signal_length < 2:
            return signal

        diff_norm = np.linalg.norm(np.diff(signal, axis=0), axis=1)
        threshold = onset_threshold_factor * np.max(diff_norm)
        idxs = np.where(diff_norm > threshold)[0]
        if idxs.size == 0:
            return signal

        last_active = idxs[-1] + 1
        end_idx = min(last_active + post_trigger_points, signal_length)
        return signal[:end_idx]

    def concatenate_finger_sequences(self, csv_path):
        df = pd.read_csv(csv_path)
        df.sort_values(by='timestamp', inplace=True)

        sensor_columns = df.columns[2:]
        raw_sensor_data = df[sensor_columns].values.astype(np.float32)
        filtered_sensor_data = self.apply_butterworth(raw_sensor_data)
        df[sensor_columns] = filtered_sensor_data

        groups = list(df.groupby(df['status'].ne(df['status'].shift()).cumsum()))
        instances = []
        labels = []

        for idx, (_, group) in enumerate(groups):
            label = group['status'].iloc[0]
            if label == 'open' or len(group) < self.min_instance_length:
                continue
            sensor_data = group.iloc[:, 2:].values.astype(np.float32)

            preceding_data = None
            if idx > 0:
                prev_label = groups[idx - 1][1]['status'].iloc[0]
                if prev_label == 'open':
                    preceding_data = groups[idx - 1][1].iloc[:, 2:].values.astype(np.float32)

            following_data = None
            if idx + 1 < len(groups) and groups[idx + 1][1]['status'].iloc[0] == 'open':
                following_data = groups[idx + 1][1].iloc[:, 2:].values.astype(np.float32)

            aligned_data, trigger_offset = self.align_onset(sensor_data, preceding_data, following_data)
            trimmed_data = self.trim_tail(aligned_data)
            instances.append(trimmed_data)

            label_array = np.full((len(trimmed_data),), label) # Renamed for clarity
            labels.append(label_array)

        if self.num_instances is not None and len(instances) > self.num_instances:
            instances = instances[:self.num_instances]
            labels = labels[:self.num_instances]
        
        instance_lengths = [len(inst) for inst in instances]
        return instances, labels, instance_lengths

    def concatenate_instances_from_folder(self, folder_path):
        hardcoded_csv_files = [
            'thumb.csv',
            'index.csv',
            'middle.csv',
            'ring.csv',
            'pinky.csv',
        ]
        csv_files = [os.path.join(folder_path, f) for f in hardcoded_csv_files if os.path.exists(os.path.join(folder_path, f))]

        if not csv_files:
            print("Warning: No specified CSV files found in the folder.")
            return np.array([]), np.array([]), [], [], np.array([])

        most_significant_idx_list = []
        all_concatenated_instances_list = [] # Stores concatenated instances from each file
        all_labels_list = [] # Stores concatenated labels from each file
        all_individual_instance_lengths = [] # Stores lengths of each individual instance across all files
        all_msi_per_row_list = [] # Stores the most significant channel index for each row, corresponding to its original file
        all_instance_idx_per_row_list = [] # Stores the instance index for each row
        global_instance_counter = 0 # Global counter for instance indices across all files

        # First pass: collect instances, labels, lengths, and determine most significant channels per file
        for file_path in csv_files:
            instances, labels_per_file, instance_lengths = self.concatenate_finger_sequences(file_path)
            if not instances:
                continue # Skip file if no instances found

            # Concatenate instances and labels for the current file
            current_file_concatenated_instances = np.concatenate(instances, axis=0)
            current_file_concatenated_labels = np.concatenate(labels_per_file, axis=0)

            # Determine most significant channel for this file's data
            variances = np.var(current_file_concatenated_instances, axis=0)
            most_significant_idx = np.argmax(variances)
            
            # Ensure uniqueness of the selected channel index across files
            if most_significant_idx in most_significant_idx_list:
                sorted_indices = np.argsort(variances)[::-1]
                for idx_val in sorted_indices:
                    if idx_val not in most_significant_idx_list:
                        most_significant_idx = idx_val
                        break
            most_significant_idx_list.append(most_significant_idx)

            # Create an array of the most significant index, one for each row from this file
            msi_for_current_file_rows = np.full(current_file_concatenated_instances.shape[0], most_significant_idx)
            all_msi_per_row_list.append(msi_for_current_file_rows)

            # Create instance indices for each row from this file
            instance_idx_for_current_file_rows = []
            for inst_length in instance_lengths:
                instance_indices = np.full(inst_length, global_instance_counter)
                instance_idx_for_current_file_rows.extend(instance_indices)
                global_instance_counter += 1
            all_instance_idx_per_row_list.append(np.array(instance_idx_for_current_file_rows))

            # Accumulate data
            all_concatenated_instances_list.append(current_file_concatenated_instances)
            all_labels_list.append(current_file_concatenated_labels)
            all_individual_instance_lengths.extend(instance_lengths)

        print("Most significant channel indices (one per file):", most_significant_idx_list)

        if not all_concatenated_instances_list:
            print("No processable instances found in any of the CSV files.")
            return np.array([]), np.array([]), [], [], np.array([]), np.array([])

        # Concatenate all data from all files
        final_all_instances_concatenated = np.concatenate(all_concatenated_instances_list, axis=0)
        final_all_labels_concatenated = np.concatenate(all_labels_list, axis=0)
        final_all_msi_per_row = np.concatenate(all_msi_per_row_list, axis=0)
        final_all_instance_idx_per_row = np.concatenate(all_instance_idx_per_row_list, axis=0)

        # Second pass: select the chosen significant channels from the combined data (for plotting)
        selected_channels_to_stack_for_plot = []
        if not most_significant_idx_list: 
             print("Warning: No significant channels identified for plotting.")
             # Return all data for CSV, but empty array for plot-specific data
             return final_all_instances_concatenated, np.array([]), final_all_labels_concatenated, all_individual_instance_lengths, final_all_msi_per_row, final_all_instance_idx_per_row

        for sig_idx in most_significant_idx_list:
            if sig_idx < final_all_instances_concatenated.shape[1]:
                selected_channel_data = final_all_instances_concatenated[:, sig_idx]
                selected_channels_to_stack_for_plot.append(selected_channel_data)
            else:
                print(f"Warning: sig_idx {sig_idx} is out of bounds for concatenated data with {final_all_instances_concatenated.shape[1]} channels. Skipping this channel for plotting.")

        plot_specific_selected_channels = np.array([])
        if selected_channels_to_stack_for_plot:
            plot_specific_selected_channels = np.column_stack(selected_channels_to_stack_for_plot)
        else:
            print("No channels were selected for plotting after significance analysis.")
            # We still return all data for CSV
            
        return final_all_instances_concatenated, plot_specific_selected_channels, final_all_labels_concatenated, all_individual_instance_lengths, final_all_msi_per_row, final_all_instance_idx_per_row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate finger sequences from CSV files in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing CSV files.')
    parser.add_argument('--num-instances', type=int, default=None, help='Number of instances to concatenate from each file.')
    args = parser.parse_args()

    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['font.size'] = 8

    processor = FingerSequenceProcessor(num_instances=args.num_instances)
    # Renamed variables for clarity based on their content/use
    all_channels_data, plot_channels_data, flat_labels, instance_lengths, msi_per_row_data, instance_idx_data = processor.concatenate_instances_from_folder(args.folder_path)


    if all_channels_data.size == 0: # Check if any data was processed at all
        print("No data to process or plot.")
    else:
        # Save all channels, labels, and the most significant channel index to a CSV file
        df_out = pd.DataFrame(all_channels_data)
        # Dynamically name columns for all original channels
        df_out.columns = [f'original_channel_{i}' for i in range(all_channels_data.shape[1])]
        df_out['label'] = flat_labels
        df_out['most_significant_channel_idx'] = msi_per_row_data
        df_out['instance_index'] = instance_idx_data
        
        num_instances_str = str(args.num_instances) if args.num_instances is not None else "all"
        # Ensure the directory exists
        output_dir = './figures/fig2/concatenated_finger_sequence/data/'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f'concatenated_all_channels_sequences_{num_instances_str}.csv')
        df_out.to_csv(output_filename, index=False)
        print(f"Saved concatenated data (all channels) to {output_filename}")

        # Plotting code remains the same, using plot_channels_data (which are the selected significant channels)
        if plot_channels_data.size == 0:
            print("No data available for plotting (selected significant channels).")
        else:
            fig, ax = plt.subplots(figsize=(6, 1.5))

            # Colors are for the plot_channels_data, which has one column per original file's most significant channel
            if plot_channels_data.shape[1] > 0:
                colors = [
                    '#8c64b4',
                    '#8cb464',
                    '#6478b4',
                    '#b4a064',
                    '#64b4b4',
                ]
                # Ensure colors list is long enough, or handle cases with more channels than colors
                if plot_channels_data.shape[1] > len(colors):
                    # Simple extension: repeat colors or use a colormap
                    import matplotlib.cm as cm
                    cmap_colors = cm.viridis(np.linspace(0, 1, plot_channels_data.shape[1]))
                    colors = cmap_colors 

            else:
                colors = [] # No channels to plot

            current_x_offset = 0
            avg_instance_len = np.mean(instance_lengths) if instance_lengths else 0
            gap_size = int(avg_instance_len * 0.1) if avg_instance_len > 0 else 20 
            if gap_size == 0 and instance_lengths : gap_size = 1 

            data_cursor = 0
            for i_idx, inst_len in enumerate(instance_lengths):
                if inst_len == 0: continue 

                # Slicing from plot_channels_data
                instance_segment = plot_channels_data[data_cursor : data_cursor + inst_len]
                x_coords = np.arange(current_x_offset, current_x_offset + inst_len)

                for ch_idx in range(instance_segment.shape[1]): 
                    channel_plot_label = f'File {ch_idx} Sig. Ch.' if i_idx == 0 else None # Clarified label
                    plt.plot(x_coords, instance_segment[:, ch_idx],
                             color=colors[ch_idx % len(colors)], label=channel_plot_label, # Use modulo for color safety
                             lw=1, alpha=0.8)
                
                data_cursor += inst_len
                current_x_offset += inst_len + gap_size
            
            if current_x_offset > 0 and gap_size > 0 : # Ensure plotting happened before setting xlim
                 plt.xlim(0, current_x_offset - gap_size)
            elif current_x_offset > 0:
                 plt.xlim(0, current_x_offset)

            plt.xticks(np.arange(0, current_x_offset, step=200))
            if any(instance_lengths) and plot_channels_data.size > 0: 
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            fig.patch.set_alpha(0)  
            ax.set_facecolor('none')  

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.tick_params(axis='x', direction='inout')
            ax.spines['bottom'].set_position(('data', 0))
            
            # Ensure the directory for saving figures exists
            figure_output_dir = 'figures/fig2/concatenated_finger_sequence/'
            os.makedirs(figure_output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_output_dir, 'krushna_plot_significant.svg'), dpi=600, bbox_inches='tight', transparent=True)
            plt.show()