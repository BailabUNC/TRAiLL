import os
import time
import datetime
import argparse
import logging
from functools import partial
from multiprocessing import Queue, Process, Event, Manager
from multiprocessing.managers import Namespace
from multiprocessing.synchronize import Event as SyncEvent

import serial
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.ndimage import zoom
from cmap import Colormap

from benchmark import traill_benchmark

mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 14
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TRAiLLVisualizer:
    def __init__(self,
                 serial_port,
                 baud_rate=115200,
                 data_folder=None,
                 timeout=0.5,
                 action_duration=100,
                 disable_csv=False):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_folder = data_folder
        self.timeout = timeout
        self.action_duration = action_duration

        self.vis_queue = Queue()
        self.saving_queue = Queue()

        self.disable_csv = disable_csv
        
        self.filepath = None
        self.fig, self.ax = None, None
        self.terminate_loop_evt = Event()
        self.shared_status = None
        
        if self.data_folder is None:
            self.data_folder = 'raw_data'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def connect(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            logging.info(f'Connected to {self.serial_port} at {self.baud_rate} baud.')
        except serial.SerialException as e:
            logging.error(f'Error connecting to serial port: {e}')
            raise

    def set_destination(self):
        if self.disable_csv:
            logging.info('Saving is disabled, no csv will be created.')
            return
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filepath = os.path.join(self.data_folder, f'test-data-{timestamp}.csv')
        header = 'timestamp,status,' + ','.join([f"ch{i+1}" for i in range(48)]) + '\n'
        with open(self.filepath, 'w') as f:
            f.write(header)
        logging.info(f'Data will be saved to {self.filepath}')
    
    def parse(self, lines: list):
        '''
        Parses a set of data lines into a 2D array
        '''
        try:
            data = [list(map(int, line.split())) for line in lines if line.strip()]
            return np.array(data)
        except ValueError as e:
            logging.error(f'Error parsing data: {e}')
            return np.zeros((6, 8))
        
    def update_img(self, frame):
        # Drain the queue to get the latest data sample
        latest_data = None
        while not self.vis_queue.empty():
            latest_data = self.vis_queue.get()
        if latest_data is not None:
            display_data = zoom(latest_data, (3, 3), order=1)
            self.img.set_data(display_data)

        current_status = self.shared_status.status
        if self.radio is not None:
            # radio.value_selected returns the currently selected option
            if self.radio.value_selected != current_status:
                try:
                    new_index = self.activities.index(current_status)
                except ValueError:
                    new_index = 0
                self.radio.set_active(new_index)
        return [self.img]
    
    def update_status(self, new_status):
        self.shared_status.status = new_status
        logging.info(f'Status updated to: {new_status}')
    
    def _serial_process(self):
        '''
        Reads data from the serial port and puts it in the queue.
        This is the child process.
        '''
        baseline = None  # store the first data sample per channel
        try:
            self.connect()
            line_buffer = []

            # Keep reading until terminate_evt is set
            while not self.terminate_loop_evt.is_set(): 
                try:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
                except Exception as decode_err:
                    logging.error(f'Decoding error: {decode_err}')
                    continue

                if line == "":
                    continue  # skip empty line

                line_buffer.append(line)
                # logging.info(line)
                if len(line_buffer) == 6:
                    data = self.parse(line_buffer)
                    if data.shape == (6, 8):
                        data = np.roll(data, -1)  # fix the data order problem
                        if baseline is None:
                            baseline = data.copy()
                        data = data - baseline
                        self.vis_queue.put(data)
                        if not self.disable_csv:
                            self.saving_queue.put(data)
                    line_buffer.clear()
        
        except Exception as e:
            logging.error(f'Error in serial process: {e}')
        except KeyboardInterrupt:
            self.terminate_loop_evt.set()
        finally:
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
                logging.info('Serial port closed.')
    
    @staticmethod
    def _saving_process(saving_queue: Queue,
                        filepath: str,
                        terminate_loop_evt: SyncEvent,
                        shared_status: Namespace,
                        action_duration: int):
        """
        Stand-alone process for saving data to disk.
        Opens the file once and continuously drains the saving queue, writing each
        matrix with a timestamp and current status. When a non-"open" action is active,
        after a pre-defined number of data points (i.e. matrices) have been saved, the status
        is automatically reset to "open."
        """
        last_ts = 0.0  # last saved timestamp in seconds

        current_action = 'open'
        points_count = 0
        try:
            with open(filepath, 'a') as f:
                while not (terminate_loop_evt.is_set() and saving_queue.empty()):
                    try:
                        data = saving_queue.get(timeout=0.1)
                    except Exception as e:  # Likely queue.Empty; simply continue looping
                        logging.error(f'Cannot acquire data: {e}')
                        continue

                    # Update counter only when a non-'open' action is active
                    if shared_status.status != 'open':
                        # If the action remains the same, increment; otherwise, reset counter
                        if shared_status.status == current_action:
                            points_count += 1
                        else:
                            current_action = shared_status.status
                            points_count = 1
                    else:
                        current_action = 'open'
                        points_count = 0

                    # If the current action has reached its pre-defined duration, reset to 'open'
                    if current_action != 'open' and points_count >= action_duration:
                        shared_status.status = 'open'
                        current_action= 'open'
                        points_count = 0

                    # Generate the current timestamp using the serial port reported time
                    now_ts = round(datetime.datetime.now().timestamp(), 2)
                    if now_ts <= last_ts:
                        now_ts = last_ts + 0.01
                    last_ts = now_ts
                    ts_dt = datetime.datetime.fromtimestamp(now_ts)
                    # Format the timestamp with exactly 2 digits after the decimal point.
                    # The %f format gives microseconds; we trim the last 4 digits.
                    timestamp = ts_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]

                    flattened_data = data.flatten()
                    status = shared_status.status
                    f.write(f'{timestamp},{status},' + ','.join(map(str, flattened_data)) + '\n')
                    f.flush()
        except Exception as e:
            logging.error(f'Error in saving process: {e}')

    def _visualization_process(self):
        '''
        Handles real-time visualization of the NIRS mapping.
        '''
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=-0.12, bottom=0.2)
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        self.img = self.ax.imshow(np.zeros((6, 8)), cmap=Colormap('cmocean:balance').to_mpl(), vmin=-200, vmax=200)
        
        ax_pos = self.ax.get_position()   # [x0, y0, width, height]
        button_height = 0.075
        ax_terminate = plt.axes([ax_pos.x0,
                                 ax_pos.y0 - button_height - 0.01,
                                 ax_pos.width,
                                 button_height])
        terminate_button = Button(ax_terminate, 'Terminate', color='red', hovercolor='lightcoral')
        terminate_button.on_clicked(self.terminate)

        # Create status buttons for the activities.
        self.activities = ['open',
                           'fist',
                           'point',
                           'pinch',
                           'wave',
                           'trigger',
                           'grab',
                           'thumbs-up',
                           'swipe']
        panel_width = 0.25
        ax_radio = plt.axes([ax_pos.x0 + ax_pos.width + 0.01,
                             ax_pos.y0,
                             panel_width,
                             ax_pos.height])
        self.radio = RadioButtons(ax_radio, self.activities, active=0)
        self.radio.on_clicked(self.update_status)

        anim = FuncAnimation(self.fig, self.update_img, interval=10,
                             cache_frame_data=False, blit=False)
        plt.show()

    def terminate(self, event):
        logging.info('Termination request by user.')
        self.terminate_loop_evt.set()
        if hasattr(self, 'serial_process') and self.serial_process.is_alive():
            self.serial_process.join()
        if hasattr(self, 'saving_process') and self.saving_process.is_alive():
            self.saving_process.terminate()
        plt.close(self.fig)
        logging.info('Test terminated by user.')

    def run(self):
        self.set_destination()

        manager = Manager()
        self.shared_status = manager.Namespace()
        self.shared_status.status = 'open'
        
        self.serial_process = Process(target=self._serial_process)
        self.serial_process.start()

        if not self.disable_csv:
            self.saving_process = Process(
                target=self._saving_process,
                args=(
                    self.saving_queue,
                    self.filepath,
                    self.terminate_loop_evt,
                    self.shared_status,
                    self.action_duration,
                )
            )
            self.saving_process.start()

        # main process
        self._visualization_process()

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
    