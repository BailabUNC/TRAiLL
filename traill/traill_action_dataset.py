# traill_dataset.py

import os
import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample
import torch
from torch.utils.data import Dataset

class TRAiLLActionDataset(Dataset):
    def __init__(self,
                 csv_path,
                 target_length=128,
                 onset_threshold_factor=0.5,
                 min_instance_length=5,
                 pre_trigger_points=15,
                 filter_order=3,
                 filter_cutoff=7,
                 fs=100,
                 transform=None):
        """
        Args:
            csv_path (str): Path to the raw CSV file.
            target_length (int): Number of time steps to resample each instance to.
            onset_threshold_factor (float): Fraction of max derivative norm used to detect the gesture onset.
            min_instance_length (int): Minimum number of rows for an instance to be considered.
            pre_trigger_points (int): Number of points to trace back before the detected onset.
            filter_order (int): Order of the Butterworth filter.
            filter_cutoff (list): Cutoff frequency (Hz) for the Butterworth filter.
            fs (float): Sampling frequency (Hz) of the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_path = csv_path
        self.target_length = target_length
        self.onset_threshold_factor = onset_threshold_factor
        self.min_instance_length = min_instance_length
        self.pre_trigger_points = pre_trigger_points
        self.filter_order = filter_order
        self.filter_cutoff = filter_cutoff
        self.fs = fs
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

        # Load and process the data file
        self.instances = self._prepare_instances()

    def _prepare_instances(self):
        # Load the csv files
        df = pd.read_csv(self.csv_path)
        df.sort_values(by='timestamp', inplace=True)

        sensor_columns = df.columns[2:]
        raw_sensor_data = df[sensor_columns].values.astype(np.float32)
        filtered_sensor_data = self._apply_butterworth(raw_sensor_data)
        df[sensor_columns] = filtered_sensor_data

        groups = list(df.groupby(df['status'].ne(df['status'].shift()).cumsum()))
        instances = []

        for idx, (group_id, group) in enumerate(groups):
            label = group['status'].iloc[0]
            if label == 'open' or len(group) < self.min_instance_length:
                continue
            sensor_data = group.iloc[:, 2:].values.astype(np.float32)

            # Check if there is a preceding group that is 'open'
            preceding_data = None
            if idx > 0:
                prev_label = groups[idx - 1][1]['status'].iloc[0]
                if prev_label == 'open':
                    preceding_data = groups[idx - 1][1].iloc[:, 2:].values.astype(np.float32)

            trimmed_data, trigger_offset = self._align_onset(sensor_data, preceding_data)

            # Resample and normalize the trimmed data.
            sensor_data_resampled, scaled_trigger = self._resample_signal(trimmed_data, trigger_offset)
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
                     btype='low', fs=self.fs, output='sos', analog=False)
        filtered_data = sosfiltfilt(sos, sensor_data, axis=0, padlen=self.target_length // 2)
        return filtered_data

    def _align_onset(self, active_data, preceding_data):
        """
        Detect the gesture onset using only the active_data.
        Then, if preceding_data is provided and the active onset occurs within 
        pre_trigger_points, prepend just enough rows from preceding_data so that
        the effective segment includes the desired pre-trigger window.
        
        Returns:
            concatenated_data: Active data with the necessary pre-trigger rows (if any).
            trigger_offset: The index in the concatenated data corresponding to the onset.
        """
        # Compute onset index in active_data.
        diff_norm = np.linalg.norm(np.diff(active_data, axis=0), axis=1)
        threshold = self.onset_threshold_factor * np.max(diff_norm)
        onset_indices = np.where(diff_norm > threshold)[0]
        if onset_indices.size > 0:
            onset_index_active = onset_indices[0]
        else:
            onset_index_active = 0
        
        if preceding_data is not None:
            # Determine how many pre-trigger points we need from the active group.
            extra_needed = self.pre_trigger_points - onset_index_active
            if extra_needed > 0:
                # Use as many rows as available from preceding_data (up to extra_needed)
                num_rows_preceding = preceding_data.shape[0]
                rows_to_prepend = preceding_data[max(0, num_rows_preceding - extra_needed):]
            else:
                rows_to_prepend = np.empty((0, active_data.shape[1]))
            concatenated_data = np.concatenate((rows_to_prepend, active_data), axis=0)
            trigger_offset = rows_to_prepend.shape[0] + onset_index_active
            return concatenated_data, trigger_offset
        else:
            # Without preceding data, simply trace back within active_data.
            effective_start = max(0, onset_index_active - self.pre_trigger_points)
            trigger_offset = onset_index_active - effective_start
            trimmed_data = active_data[effective_start:]
            return trimmed_data, trigger_offset

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
    parser.add_argument('test_name', type=str,
                        help='Name of the data csv file to process.')
    args = parser.parse_args()
    path = os.path.join('data', args.person, args.test_name + '.csv')

    dataset = TRAiLLActionDataset(path)

    print(f'Visualizing {len(dataset)} instances...')

    features_list = []
    for features, _ in dataset:
        features_list.append(features.numpy())
    trigger_indices = [inst['trigger_index'] for inst in dataset.instances]
    all_features = np.stack(features_list, axis=0)  # shape: (num_instances, target_length, num_channels)
    num_instances, target_length, num_channels = all_features.shape

    import matplotlib.pyplot as plt

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
        ax.set_ylim([-2, 2])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()