# traill_action_dataset.py

import os
import argparse
import glob
import string

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class TRAiLLDataset(Dataset):
    def __init__(self,
                 csv_path,
                 target_length=128,
                 onset_threshold_factor=0.5,
                 min_instance_length=5,
                 pre_trigger_points=15,
                 post_trigger_points=15,
                 filter_order=3,
                 filter_cutoff=[0.2, 30],
                 fs=100,
                 apply_filter=True, # New parameter
                 transform=None):
        """
        Args:
            csv_path (str): Path to the raw CSV file.
            target_length (int): Number of time steps to resample each instance to.
            onset_threshold_factor (float): Fraction of max derivative norm used to detect the gesture onset.
            min_instance_length (int): Minimum number of rows for an instance to be considered.
            pre_trigger_points (int): Number of points to trace back before the detected onset.
            post_trigger_points (int): Number of points to trace back after the gesture.
            filter_order (int): Order of the Butterworth filter.
            filter_cutoff (list): Cutoff frequency (Hz) for the Butterworth filter.
            fs (float): Sampling frequency (Hz) of the data.
            apply_filter (bool): Whether to apply Butterworth filter to the feature data. Filtered data is always used for indexing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_path = csv_path
        self.target_length = target_length
        self.onset_threshold_factor = onset_threshold_factor
        self.min_instance_length = min_instance_length
        self.pre_trigger_points = pre_trigger_points
        self.post_trigger_points = post_trigger_points
        self.filter_order = filter_order
        self.filter_cutoff = filter_cutoff
        self.fs = fs
        self.apply_filter = apply_filter # Store the new parameter
        self.transform = transform

        # Define a mapping for gesture labels
        self.label_map = {
            'fist': 1,
            'point': 2,
            'pinch': 3,
            'wave': 4,
            'trigger': 5,
            'grab': 6,
            'thumbs-up': 7,
            'swipe': 8,
        }
        # Add lowercase letters for gesture labels
        self.label_map.update({k: v for k, v in zip(string.ascii_lowercase, range(9, 35))})

        # Load and process the data file
        self.instances = self._prepare_instances()
        self.instances = self._filter_outliers(std_factor=2.8)

    def _prepare_instances(self):
        # Load the csv files
        df = pd.read_csv(self.csv_path)
        df.sort_values(by='timestamp', inplace=True)

        sensor_columns = df.columns[2:]
        raw_sensor_data_matrix = df[sensor_columns].values.astype(np.float32)
        
        # Filtered data is always created for indexing logic (onset, trim)
        filtered_sensor_data_matrix_for_indexing = self._apply_butterworth(raw_sensor_data_matrix.copy())

        # groups are based on the original df's status column
        groups = list(df.groupby(df['status'].ne(df['status'].shift()).cumsum()))
        instances = []

        for idx, (group_id, group_df) in enumerate(groups):
            label = group_df['status'].iloc[0]
            if label == 'open' or len(group_df) < self.min_instance_length:
                continue
            
            current_group_indices = group_df.index

            # Data for indexing logic (always filtered)
            active_data_for_indexing = filtered_sensor_data_matrix_for_indexing[current_group_indices]
            # Raw data (potentially for features)
            active_data_for_features_raw = raw_sensor_data_matrix[current_group_indices]

            preceding_data_for_indexing, preceding_data_for_features_raw = None, None
            if idx > 0:
                prev_group_df = groups[idx - 1][1]
                if prev_group_df['status'].iloc[0] == 'open':
                    prev_group_indices = prev_group_df.index
                    preceding_data_for_indexing = filtered_sensor_data_matrix_for_indexing[prev_group_indices]
                    preceding_data_for_features_raw = raw_sensor_data_matrix[prev_group_indices]
            
            following_data_for_indexing, following_data_for_features_raw = None, None
            if idx + 1 < len(groups) and groups[idx + 1][1]['status'].iloc[0] == 'open':
                following_group_df = groups[idx + 1][1]
                following_group_indices = following_group_df.index
                following_data_for_indexing = filtered_sensor_data_matrix_for_indexing[following_group_indices]
                following_data_for_features_raw = raw_sensor_data_matrix[following_group_indices]

            # Align onset using filtered data for logic, and apply same slicing to raw data
            aligned_data_filt, trigger_offset, aligned_data_raw_maybe = self._align_onset(
                active_data_for_indexing, preceding_data_for_indexing, following_data_for_indexing,
                active_data_alt=active_data_for_features_raw,
                preceding_data_alt=preceding_data_for_features_raw,
                following_data_alt=following_data_for_features_raw
            )

            # Trim tail using filtered data for logic, and apply same slicing to raw data
            trimmed_data_filt, trimmed_data_raw_maybe = self._trim_tail(
                aligned_data_filt,
                signal_alt=aligned_data_raw_maybe
            )

            # Select data for features based on the flag
            if self.apply_filter:
                final_data_for_features = trimmed_data_filt
            else:
                final_data_for_features = trimmed_data_raw_maybe
                if final_data_for_features is None: # Should not happen if raw inputs were provided
                    final_data_for_features = trimmed_data_filt


            # Resample and normalize the selected data.
            sensor_data_resampled, scaled_trigger = self._resample_signal(final_data_for_features, trigger_offset)
            sensor_data_normalized = self._normalize_signal(sensor_data_resampled)

            instances.append({
                'features': sensor_data_normalized,
                'label': label,
                'trigger_index': scaled_trigger
            })
        
        return instances

    def _apply_butterworth(self, sensor_data):
        """
        Apply a butterworth filter to the sensor data.
        """
        sos = butter(self.filter_order, self.filter_cutoff,
                     btype='bandpass', fs=self.fs, output='sos', analog=False)
        filtered_data = sosfiltfilt(sos, sensor_data, axis=0, padlen=self.target_length // 2)
        return filtered_data

    def _align_onset(self, active_data, preceding_data, following_data,
                     active_data_alt=None, preceding_data_alt=None, following_data_alt=None):
        """
        Detect the gesture onset using only the active_data (primary, filtered).
        Then, if preceding_data is provided and the active onset occurs within 
        pre_trigger_points, prepend just enough rows from preceding_data so that
        the effective segment includes the desired pre-trigger window.
        Applies identical slicing/concatenation to _alt data if provided.
        
        Returns:
            concatenated_data: Active data with the necessary pre-trigger rows (if any).
            trigger_offset: The index in the concatenated data corresponding to the onset.
            concatenated_data_alt: Alternative data processed identically, or None.
        """
        # Compute onset index in active_data.
        diff_norm = np.linalg.norm(np.diff(active_data, axis=0), axis=1)
        threshold = self.onset_threshold_factor * np.max(diff_norm) if diff_norm.size > 0 else 0
        onset_indices = np.where(diff_norm > threshold)[0]
        onset_index_active = int(onset_indices[0]) if onset_indices.size > 0 else 0
        
        concatenated_data = None
        concatenated_data_alt = None
        trigger_offset = 0

        # Helper to apply slices and concatenate
        def _apply_slicing_and_concat(main_pre, main_active, main_fol,
                                      alt_pre, alt_active, alt_fol,
                                      pre_slice, active_slice, fol_slice):
            parts_main = []
            parts_alt = []
            
            if pre_slice and main_pre is not None:
                parts_main.append(main_pre[pre_slice])
                if alt_pre is not None: parts_alt.append(alt_pre[pre_slice])
            
            parts_main.append(main_active[active_slice])
            if alt_active is not None: parts_alt.append(alt_active[active_slice])

            if fol_slice and main_fol is not None:
                parts_main.append(main_fol[fol_slice])
                if alt_fol is not None: parts_alt.append(alt_fol[fol_slice])
            
            concat_main = np.concatenate(parts_main, axis=0) if parts_main else np.array([]).reshape(0, main_active.shape[1] if main_active.ndim > 1 and main_active.shape[1] > 0 else 0)

            concat_alt = None
            if alt_active is not None: # If primary alt data is present, attempt concatenation
                # Ensure alt_parts only contains non-None elements if their main counterparts were added
                valid_alt_parts = []
                if pre_slice and main_pre is not None and alt_pre is not None: valid_alt_parts.append(alt_pre[pre_slice])
                valid_alt_parts.append(alt_active[active_slice])
                if fol_slice and main_fol is not None and alt_fol is not None: valid_alt_parts.append(alt_fol[fol_slice])

                if valid_alt_parts:
                    concat_alt = np.concatenate(valid_alt_parts, axis=0)
                elif alt_active is not None: # Handle case where only active_alt is present
                     concat_alt = alt_active[active_slice]


            return concat_main, concat_alt

        if preceding_data is not None:
            extra_needed = self.pre_trigger_points - onset_index_active
            if extra_needed > 0:
                num_rows_preceding = preceding_data.shape[0]
                num_to_prepend = min(extra_needed, num_rows_preceding)
                
                pre_s = slice(num_rows_preceding - num_to_prepend, None) if num_to_prepend > 0 else None
                active_s = slice(None) # Full active data

                concatenated_data, concatenated_data_alt = _apply_slicing_and_concat(
                    preceding_data, active_data, None,
                    preceding_data_alt, active_data_alt, None,
                    pre_s, active_s, None
                )
                trigger_offset = num_to_prepend + onset_index_active
            else: # extra_needed <= 0
                num_rows_to_trim_active_start = -extra_needed
                
                active_s = slice(num_rows_to_trim_active_start, None)
                fol_s = None
                
                if following_data is not None and num_rows_to_trim_active_start > 0:
                    num_to_append_following = min(num_rows_to_trim_active_start, following_data.shape[0])
                    fol_s = slice(0, num_to_append_following) if num_to_append_following > 0 else None

                concatenated_data, concatenated_data_alt = _apply_slicing_and_concat(
                    None, active_data, following_data,
                    None, active_data_alt, following_data_alt,
                    None, active_s, fol_s
                )
                trigger_offset = self.pre_trigger_points
        else: # No preceding data
            effective_start = max(0, onset_index_active - self.pre_trigger_points)
            active_s = slice(effective_start, None)
            
            concatenated_data, concatenated_data_alt = _apply_slicing_and_concat(
                None, active_data, None,
                None, active_data_alt, None,
                None, active_s, None
            )
            trigger_offset = onset_index_active - effective_start
            
        return concatenated_data, trigger_offset, concatenated_data_alt
        
    def _trim_tail(self, signal, signal_alt=None):
        """
        Trim off any trailing "rest" after the gesture.
        Uses only past data (no peeking ahead), by finding
        the last index where the derivative norm exceeds 
        `onset_threshold_facter * max_derivative`.
        Applies identical trimming to signal_alt if provided.
        """
        signal_length = signal.shape[0]

        if signal_length < 2:
            return signal, signal_alt
        
        diff_norm = np.linalg.norm(np.diff(signal, axis=0), axis=1)
        if diff_norm.size == 0: # Should not happen if signal_length >= 2
             return signal, signal_alt

        threshold = self.onset_threshold_factor * np.max(diff_norm)
        idxs = np.where(diff_norm > threshold)[0]

        if idxs.size == 0:
            return signal, signal_alt

        last_active = idxs[-1] + 1
        end_idx = min(last_active + self.post_trigger_points, signal_length)
        
        trimmed_signal = signal[:end_idx]
        trimmed_signal_alt = None
        if signal_alt is not None:
            # Ensure signal_alt is at least as long as the portion of signal we're keeping
            # or can be sliced up to its own length if shorter.
            alt_end_idx = min(end_idx, signal_alt.shape[0])
            trimmed_signal_alt = signal_alt[:alt_end_idx]
            # If signal_alt was shorter than signal, and end_idx was beyond signal_alt's length,
            # this correctly takes up to signal_alt's end.
            # If signal_alt was longer, it's trimmed like signal.

        return trimmed_signal, trimmed_signal_alt

    def _resample_signal(self, signal, trigger_offset):
        """
        Resample the signal to target_length and scale the trigger offset accordingly.
        Returns the resampled signal and the scaled trigger offset.
        """
        original_length = signal.shape[0]
        resampled_signal = resample(signal, self.target_length, axis=0)
        scaled_trigger = int(trigger_offset / original_length * self.target_length)
        return resampled_signal, scaled_trigger
    
    def _normalize_signal(self, signal):
        """
        Normalize the entire signal using the mean and standard deviation
        of the most significant channel (i.e., the channel with the highest variance).
        This ensures that all channels are scaled using the same factors,
        preserving the contrast between channels.
        """
        variances = np.var(signal, axis=0)

        ref_channel = np.argmax(variances)
        ref_mean = np.mean(signal[:, ref_channel])
        ref_std = np.std(signal[:, ref_channel]) + 1e-6
        normalized_signal = (signal - ref_mean) / ref_std

        return normalized_signal
    
    def _filter_outliers(self, std_factor=3):
        """
        Remove any instance whose feature at any (timestep, channel)
        falls outside [mean-std, mean+std] computed across all instances.
        """
        if self.instances is None or len(self.instances) == 0:
            return
        # stack to shape (N_instances, T, C)
        feats = np.stack([inst['features'] for inst in self.instances], axis=0)
        # mean/std at each (t, c)
        means = feats.mean(axis=0)
        stds  = feats.std(axis=0)

        filtered = []
        for inst in self.instances:
            f = inst['features']  # shape (T, C)
            # keep only if every point in every channel is within mean±std
            inlier_mask = (f >= (means - std_factor * stds)) & (f <= (means + std_factor * stds))
            if inlier_mask.all():
                filtered.append(inst)
        return filtered

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        features = instance['features']  # shape: (target_length, num_channels)
        label_str = instance['label']
        label = self.label_map.get(label_str, -1)

        if self.transform:
            features = self.transform(features)

        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return features, label
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('person', type=str,
                        help='Name of the participant.')
    parser.add_argument('test_path', nargs='+',
                        help='Path components (without .csv) under data/<person> to the CSV file, e.g. subfolder session1.')
    parser.add_argument('--group', default=None, type=int,
                        help='Group name to filter the dataset by. If not set, there is only one group of data.')
    parser.add_argument('--target-length', default=512, type=int,
                        help='Number of time steps to resample each instance to.')
    parser.add_argument('--onset-threshold', default=0.5, type=float,
                        help='Fraction of max derivative norm used to detect the gesture onset.')
    parser.add_argument('--pre-trigger-points', default=15, type=int,
                        help='Number of points to trace back before the detected onset.')
    parser.add_argument('--disable-filter', action='store_false', dest='apply_filter',
                        help='If set, do not apply Butterworth filter to feature data (filtered data still used for indexing).')
    parser.add_argument('--batch', action='store_true',
                        help='If set, treat test_path as a folder and process all CSV files in it.')
    parser.add_argument('--no-plot', action='store_true',
                        help='If set, do not plot the dataset.')
    parser.add_argument('--no-save', action='store_true',
                        help='If set, do not save the dataset.')
    args = parser.parse_args()

    person_dir = os.path.join('data', args.person)

    test_dir = os.path.join(*args.test_path)
    if args.group:
        test_dir = f'{test_dir}_group_{args.group}'
        
    if args.batch:
        data_dir = os.path.join(person_dir, test_dir)
        data_files = glob.glob(os.path.join(data_dir, '*.csv'))
    else:
        data_files = [os.path.join(person_dir, test_dir + '.csv')]

    for csv in data_files:
        # derive relative path under data/<person>
        csv_path = os.path.relpath(csv, person_dir)
        csv_path_no_text = os.path.splitext(csv_path)[0]

        print(f'Processing {csv}...')
        # Load, process, and save the dataset
        dataset = TRAiLLDataset(csv, target_length=args.target_length, 
                                onset_threshold_factor=args.onset_threshold, 
                                pre_trigger_points=args.pre_trigger_points,
                                apply_filter=args.apply_filter) # Pass the new argument
        # Skip saving if --no-save is set
        if not args.no_save:
            group_str = f'-group_{args.group}' if args.group else ''
            filter_str = '' if args.apply_filter else '-no_filter'
            out_path = os.path.join('data', '.processed', f'dataset-{args.person}-{csv_path_no_text.split("\\")[-1]}{group_str}{filter_str}.pt')
            os.makedirs(os.path.join('data', '.processed'), exist_ok=True)
            print(f'Saving dataset to {out_path}...')
            torch.save(dataset, out_path)
            print(f'[{csv_path_no_text}] → {out_path} ({len(dataset)} instances)')

        else:
            print(f'Skipping save for {csv}...')

        print(f'Visualizing {len(dataset)} instances...')

        features_list = [f.numpy() for f, _ in dataset]
        trigger_indices = [inst['trigger_index'] for inst in dataset.instances]
        all_features = np.stack(features_list, axis=0)  # shape: (num_instances, target_length, num_channels)
        num_instances, target_length, num_channels = all_features.shape

        if args.no_plot:
            continue
        
        fig, axes = plt.subplots(nrows=6, ncols=8, figsize=(16, 12))
        for channel, ax in enumerate(axes.flat):
            for i in range(num_instances):
                ax.plot(all_features[i, :, channel], c='lightgray', alpha=0.2)
            
                trig_idx = trigger_indices[i]
                y_val = all_features[i, trig_idx, channel] if trig_idx < target_length else np.nan
                # ax.plot(trig_idx, y_val, marker='o', color='g', ms=6)

            avg_curve = np.mean(all_features[:, :, channel], axis=0)
            ax.plot(avg_curve, c='red', linewidth=2)
            
            ax.set_xlim([0, dataset.target_length - 1])
            ax.set_ylim([-3, 3])
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle(csv_path_no_text, fontsize=16)

        plt.show()