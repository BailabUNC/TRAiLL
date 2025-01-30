import os
import sys
import datetime
from multiprocessing import Process, Queue, Event

import serial
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np

class TRAiLLVisualizer:
    def __init__(self, serial_port='COM18', baud_rate=115200, data_folder='data', timeout=0.5):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_folder = data_folder
        self.timeout = timeout
        
        self.fig, self.ax = None, None

        self.terminate_loop_evt = Event()
        
        self.filepath = None
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def connect(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            print(f'Connected to {self.serial_port} at {self.baud_rate} baud.')
        except serial.SerialException as e:
            print(f'Error connecting to serial port: {e}')
            raise

    def set_destination(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.filepath = os.path.join(self.data_folder, f'test-data-{timestamp}.csv')
        with open(self.filepath, 'w') as f:
            f.write('timestamp,' + ','.join([f"ch{i+1}" for i in range(36)]) + '\n')
        print(f"Data will be saved to {self.filepath}")

    def parse(self, lines):
        '''
        Parses a set of data lines into a 2D array
        '''
        try:
            data = [list(map(int, line.split())) for line in lines if line.strip()]
            return np.array(data)
        except ValueError as e:
            print(f'Error parsing data: {e}')
            return np.zeros((6, 6))

    def save(self, data):
        if self.filepath is None:
            raise RuntimeError('File path is not set. Call set_destination() fisrt.')
        flattened_data = data.flatten()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        with open(self.filepath, 'a') as f:
            f.write(f'{timestamp},' +  ','.join(map(str, flattened_data)) + '\n')

    def update(self, frame):
        if not self.data_queue.empty():
            data = self.data_queue.get()
            self.img.set_data(data)
        return [self.img]
    
    def _serial_process(self):
        '''
        Reads data from the serial port and puts it in the queue.
        This is the child process.
        '''
        try:
            self.connect()
            self.set_destination()
            buffer = []

            # Keep reading until terminate_evt is set
            while not self.terminate_loop_evt.is_set():               
                line = self.ser.readline().decode('utf-8').strip()
                buffer.append(line)
                if len(buffer) == 6:
                    data = self.parse(buffer)
                    if data.shape == (6, 6):
                        self.data_queue.put(data)
                        self.save(data)
                    buffer.clear()
        
        except Exception as e:
            print(f'Error in serial process: {e}')
        except KeyboardInterrupt:
            self.terminate_loop_evt.set()
        finally:
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
    
    def _visualization_process(self):
        '''
        Animation of the NIRS mapping.
        This is the main process.
        '''
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(np.zeros((6, 6)), cmap='hot', vmin=2000, vmax=3000)
        
        ax_button = plt.axes([0.7, 0.01, 0.15, 0.05])
        terminate_button = Button(ax_button, 'Terminate', color='red', hovercolor='lightcoral')
        terminate_button.on_clicked(self.terminate)

        anim = FuncAnimation(self.fig, self.update, interval=20,
                             cache_frame_data=False, blit=True)
        print(f'Parent: {self.terminate_loop_evt.is_set()}')
        plt.show()

    def terminate(self, event):
        print('Test terminated by user.')
        self.terminate_loop_evt.set()
        if hasattr(self, 'serial_process') and self.serial_process.is_alive():
            self.serial_process.terminate()
            self.serial_process.join()
        plt.close(self.fig)

    def run(self):
        self.data_queue = Queue()
        self.serial_process = Process(target=self._serial_process)
        self.serial_process.start()

        # main process
        self._visualization_process()

if __name__ == '__main__':
    visualizer = TRAiLLVisualizer(serial_port='COM18', baud_rate=115200, timeout=None)
    visualizer.run()
        