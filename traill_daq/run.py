import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traill_visualizer import TRAiLLVisualizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TRAiLL visualizer')
    parser.add_argument('--port', type=str, default='COM31', help='Port to connect to TRAiLL.')
    parser.add_argument('--path', type=str, default=None, help='Path to save data.')
    parser.add_argument('--disable-csv', action='store_true', help='Disable csv file saving.')
    parser.add_argument('--profile', type=str, default=None, help='Name of the activity profile to use (from activity_profiles.json).')
    parser.add_argument('--action_duration', type=int, default=150, help='Number of data points to record per action.')

    args = parser.parse_args()
    serial_port = args.port
    path = args.path
    disable_csv = args.disable_csv
    profile = args.profile
    action_duration = args.action_duration

    visualizer = TRAiLLVisualizer(
        serial_port=serial_port,
        data_folder=path,
        disable_csv=disable_csv,
        profile_name=profile,
        profile_json_path='activity_profiles.json',
        action_duration=action_duration
    )
    visualizer.run()