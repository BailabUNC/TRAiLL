import os
import datetime
import argparse
import logging
from multiprocessing import Queue, Process, Event
from multiprocessing.synchronize import Event as SyncEvent

import serial
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np

from benchmark import traill_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TRAiLLVisualizer:
    def __init__(self, serial_port='COM18', baud_rate=115200, data_folder=None, timeout=0.5):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_folder = data_folder
        self.timeout = timeout

        self.vis_queue = Queue()
        self.saving_queue = Queue()
        
        self.filepath = None
        self.fig, self.ax = None, None
        self.terminate_loop_evt = Event()
        
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.filepath = os.path.join(self.data_folder, f'test-data-{timestamp}.csv')
        with open(self.filepath, 'w') as f:
            f.write('timestamp,' + ','.join([f"ch{i+1}" for i in range(36)]) + '\n')
        logging.info(f"Data will be saved to {self.filepath}")
    
    def parse(self, lines: list):
        '''
        Parses a set of data lines into a 2D array
        '''
        try:
            data = [list(map(int, line.split())) for line in lines if line.strip()]
            return np.array(data)
        except ValueError as e:
            logging.error(f'Error parsing data: {e}')
            return np.zeros((6, 6))
        
    def update(self, frame):
        # Drain the queue to get the latest data sample
        latest_data = None
        while not self.vis_queue.empty():
            latest_data = self.vis_queue.get()
        if latest_data is not None:
            self.img.set_data(latest_data)
        return [self.img]
    
    def _serial_process(self):
        '''
        Reads data from the serial port and puts it in the queue.
        This is the child process.
        '''
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
                if len(line_buffer) == 6:
                    data = self.parse(line_buffer)
                    if data.shape == (6, 6):
                        self.vis_queue.put(data)
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
                        terminate_loop_evt: SyncEvent):
        '''
        Stand-alone process for saving data to disk.
        Opens the file once and continuously drains the saving queue, writing each
        matrix with a timestamp to disk.
        '''
        try:
            with open(filepath, 'a') as f:
                while not (terminate_loop_evt.is_set() and saving_queue.empty()):
                    try:
                        data = saving_queue.get(timeout=0.1)
                    except Exception:  # Likely queue.Empty; simply continue looping
                        continue
                    flattened_data = data.flatten()
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    f.write(f'{timestamp},' + ','.join(map(str, flattened_data)) + '\n')
                    f.flush()
        except Exception as e:
            logging.error(f'Error in saving process: {e}')

    def _visualization_process(self):
        '''
        Handles real-time visualization of the NIRS mapping.
        '''
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(np.zeros((6, 6)), cmap='hot', vmin=0, vmax=2800)
        
        ax_button = plt.axes([0.7, 0.01, 0.15, 0.05])
        terminate_button = Button(ax_button, 'Terminate', color='red', hovercolor='lightcoral')
        terminate_button.on_clicked(self.terminate)

        anim = FuncAnimation(self.fig, self.update, interval=10,
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
        
        self.serial_process = Process(target=self._serial_process)
        self.serial_process.start()

        self.saving_process = Process(target=self._saving_process,
                                      args=(self.saving_queue, self.filepath, self.terminate_loop_evt))
        self.saving_process.start()

        # main process
        self._visualization_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TRAiLL visualizer')
    parser.add_argument('--port', type=str, help='Port to connect to TRAiLL.')
    parser.add_argument('--path', type=str, default=None, help='Path to save data.')
    
    args = parser.parse_args()
    serial_port = args.port
    path = args.path
    
    visualizer = TRAiLLVisualizer(serial_port=serial_port, data_folder=path)
    visualizer.run()
    