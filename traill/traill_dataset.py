# traill_dataset.py

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample
import torch
from torch.utils.data import Dataset

class TRAiLLDataset(Dataset):
    def __init__(self,
                 csv_path,
                 target_length=64,
                 onset_threshold_factor=0.5,
                 min_instance_length=5,
                 pre_trigger_points=3,
                 filter_order=2,
                 filter_cutoff=[0.2, 10],
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
        groups = df.groupby(df['status'].ne(df['status'].shift()).cumsum())

        instances = []
        for _, group in groups:
            label = group['status'].iloc[0]
            if label == 'open' or len(group) < self.min_instance_length:
                continue
            sensor_data = group.iloc[:, 2:].values.astype(np.float32)
            sensor_data = self._apply_butterworth(sensor_data)
            sensor_data = self._align_onset(sensor_data)

            # Resample the sensor data to a fixed length on time axis
            sensor_data = resample(sensor_data, self.target_length, axis=0)

            # Normalize: per-channel z-score normalization
            mean = sensor_data.mean(axis=0)
            std = sensor_data.std(axis=0) + 1e-6  # avoid divided by zero
            sensor_data = (sensor_data - mean) / std

            instances.append({
                'features': sensor_data,
                'label': label
            })
        
        return instances

    def _apply_butterworth(self, sensor_data):
        """
        Apply a butterworth filter to the sensor data.
        Filtering is performed on each channel independently using zero-phase filtering.
        """
        sos = butter(self.filter_order, self.filter_cutoff,
                     btype='bandpass', fs=self.fs, output='sos', analog=False)
        filtered_data = sosfiltfilt(sos, sensor_data, axis=0)
        return filtered_data

    def _align_onset(self, sensor_data):
        """
        A simple onset detection by computing the norm of the difference between consecutive samples.
        The first index where the derivative exceeds a fraction (onset_threshold_factor) of the maximum 
        derivative norm is chosen as the onset.
        """
        # Compute the Euclidean norm of the first difference (derivative) at each time step
        diff_norm = np.linalg.norm(np.diff(sensor_data, axis=0), axis=0)
        threshold = self.onset_threshold_factor * np.max(diff_norm)

        # Find the first index where the difference exceeds the threshold
        onset_indices = np.where(diff_norm > threshold)[0]
        if onset_indices.size > 0:
            onset_index = onset_indices[0]
            # Trace back a few points, ensuring we do not go below index 0.
            onset_index = max(0, onset_index - self.pre_trigger_points)
            sensor_data = sensor_data[onset_index:]
        return sensor_data
    
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

    dataset = TRAiLLDataset(path)

    features_list = []
    for features, _ in dataset:
        features_list.append(features.numpy())
    all_features = np.stack(features_list, axis=0)  # shape: (n_instances, target_length, num_channels)
    num_instances, target_length, num_channels = all_features.shape

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=6, ncols=8, figsize=(16, 12))
    for channel, ax in enumerate(axes.flat):
        for i in range(num_instances):
            ax.plot(all_features[i, :, channel], c='lightgray', alpha=0.2)
        
        avg_curve = np.mean(all_features[:, :, channel], axis=0)
        ax.plot(avg_curve, c='red', linewidth=2)
        
        ax.set_xlim([0, 63])
        ax.set_ylim([-3, 3])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()