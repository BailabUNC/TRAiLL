import os
import datetime
from multiprocessing import Process, Queue

import serial
import matplotlib.pyplot as plt
import numpy as np

class TRAiLLVisualizer:
    def __init__(self, serial_port='COM18', baud_rate=115200, data_folder='data'):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_folder = data_folder
        self.buffer = []

        self.ser = None
        self.fig, self.ax, self.img = None, None, None
        
        self.filepath = None

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def connect(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f'Connected to {self.serial_port} at {self.baud_rate} baud.')
        except serial.SerialException as e:
            print(f'Error connecting to serial port: {e}')
            raise

    def initialize(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(np.zeros((6, 6)), cmap='viridis', vmin=500, vmax=2000)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.data_folder, f'test_{timestamp}.csv')
        print(f'Starting a new test session. Data will be saved to {self.filepath}')
        
    def parse(self, lines):
        try:
            data = [list(map(int, line.split())) for line in lines if line.strip()]
            return np.array(data)
        except ValueError as e:
            print(f'Error parsing data: {e}')
            return np.zeros((6, 6))
        
    def update(self, data):
        self.img.set_data(data)
        self.ax.set_title("TRAiLL")
        plt.pause(0.1)

    def save(self, data):
        if self.filepath is None:
            self.start_new_test()

        flatten_data = data.flatten().reshape(1, -1)
        with open(self.filepath, 'a') as file:
            np.savetxt(file, flatten_data, delimiter=',', fmt='%d')
        print(f'Appended data to {self.filepath}')

    def run(self):
        try:
            self.connect()
            self.initialize()

            while True:
                try:
                    line = self.ser.readline().decode('utf-8').strip()

                    if line == '':
                        if self.buffer:
                            data = self.parse(self.buffer)
                            if data.shape == (6, 6):
                                self.update(data)
                                self.save(data)
                            else:
                                continue
                            self.buffer.clear()
                    else:
                        self.buffer.append(line)
                except KeyboardInterrupt:
                    print('Exiting...')
                    break
                except Exception as e:
                    print(f'Error: {e}')
                    continue
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
            plt.close()

if __name__ == '__main__':
    visualizer = TRAiLLVisualizer(serial_port='COM18', baud_rate=115200)
    visualizer.run()
