from DAQ.traill_visualizer import TRAiLLVisualizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TRAiLL visualizer')
    parser.add_argument('--port', type=str, default='COM31', help='Port to connect to TRAiLL.')
    parser.add_argument('--path', type=str, default=None, help='Path to save data.')
    parser.add_argument('--disable-csv', action='store_true', help='Disable csv file saving.')
    
    args = parser.parse_args()
    serial_port = args.port
    path = args.path
    disable_csv = args.disable_csv
    
    visualizer = TRAiLLVisualizer(serial_port=serial_port, data_folder=path, disable_csv=disable_csv)
    visualizer.run()