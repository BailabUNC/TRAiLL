import os
import time
import datetime
import logging
import json
from threading import Thread
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

from traill_daq.benchmark import traill_benchmark

mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['font.size'] = 14
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TRAiLLVisualizer:
    def __init__(self,
                 serial_port,
                 baud_rate=115200,
                 data_folder=None, 
                 timeout=0.5,
                 action_duration=None,  # changed default to None
                 disable_csv=False,
                 profile_name=None,
                 profile_json_path='activity_profiles.json',
                 paper_tape_name=None,
                 paper_tape_json_path='paper_tapes.json',
                 start_delay=0,
                 auto_zero_interval=None):  # NEW: seconds between automatic re-zero, None disables
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.data_folder = data_folder
        self.timeout = timeout

        self.vis_queue = Queue()
        self.saving_queue = Queue()

        self.disable_csv = disable_csv
        # NEW: store interval (None or <=0 disables)
        self.auto_zero_interval = auto_zero_interval if (auto_zero_interval is not None and auto_zero_interval > 0) else None
        
        self.filepath = None
        self.fig, self.ax = None, None
        self.terminate_loop_evt = Event()
        self.shared_status = None

        # Always resolve the JSON paths relative to this script's directory
        self.profile_name = profile_name
        self.profile_json_path = os.path.join(os.path.dirname(__file__), profile_json_path)
        self.paper_tape_name = paper_tape_name
        self.paper_tape_json_path = os.path.join(os.path.dirname(__file__), paper_tape_json_path)
        self.start_delay = start_delay

        if self.data_folder is None:
            self.data_folder = 'raw_data'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # Load activities and action_duration from the shared JSON file if profile_name is provided
        self.action_duration = 150  # default fallback
        if self.profile_name is not None and os.path.exists(self.profile_json_path):
            with open(self.profile_json_path, 'r') as f:
                profiles = json.load(f)
            if self.profile_name in profiles:
                profile = profiles[self.profile_name]
                acts = profile.get('activities', [])
                # Ensure "open" is always the first/default status
                self.activities = ['open'] + [a for a in acts if a != 'open']
                # Load action_duration from profile if present
                if 'action_duration' in profile:
                    self.action_duration = profile['action_duration']
                elif action_duration is not None:
                    self.action_duration = action_duration
                logging.info(f'Loaded activities for profile "{self.profile_name}": {self.activities}')
                logging.info(f'Action duration for profile "{self.profile_name}": {self.action_duration}')
            else:
                self.activities = ['open']
        else:
            self.activities = ['open']  # fallback if no profile
            if action_duration is not None:
                self.action_duration = action_duration
        # Load paper tape sequence if provided
        self.tape_activities = None
        if self.paper_tape_name is not None and os.path.exists(self.paper_tape_json_path):
            try:
                with open(self.paper_tape_json_path, 'r') as f:
                    tapes = json.load(f)
                tape = tapes.get(self.paper_tape_name, None)
                if tape:
                    acts = tape.get('activities', [])
                    durs = tape.get('durations', [])
                    interval = tape.get('interval', 0)
                    self.start_delay = tape.get('start_delay', self.start_delay)
                    if len(acts) == len(durs):
                        self.tape_activities = acts
                        self.tape_durations = durs
                        self.tape_interval = interval
                        logging.info(f'Loaded paper tape "{self.paper_tape_name}": {acts} with durations {durs}, interval {interval}, and start_delay {self.start_delay}')
                    else:
                        logging.error('Paper tape activities and durations length mismatch, disabling paper tape.')
                        self.paper_tape_name = None
                else:
                    logging.error(f'Paper tape "{self.paper_tape_name}" not found, disabling paper tape.')
                    self.paper_tape_name = None
            except Exception as e:
                logging.error(f'Error loading paper tape: {e}')
                self.paper_tape_name = None

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
        last_zero_time = time.time()
        zero_count = 0
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
                    if data.shape == (6, 8):
                        data = np.roll(data, -1)  # fix the data order problem
                        if baseline is None:
                            baseline = data.copy()
                            last_zero_time = time.time()
                            logging.info('Initial baseline captured.')
                        # Periodic auto-zero
                        elif self.auto_zero_interval is not None and (time.time() - last_zero_time) >= self.auto_zero_interval:
                            baseline = data.copy()
                            last_zero_time = time.time()
                            zero_count += 1
                            logging.info(f'Auto-zero #{zero_count}: baseline refreshed.')
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
                        action_duration: int,
                        paper_tape_mode: bool = False):
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
                    if not paper_tape_mode and current_action != 'open' and points_count >= action_duration:
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
        self.img = self.ax.imshow(np.zeros((6, 8)), cmap=Colormap('cmocean:balance').to_mpl(), vmin=-2000, vmax=2000)
        
        ax_pos = self.ax.get_position()   # [x0, y0, width, height]
        button_height = 0.075
        ax_terminate = plt.axes([ax_pos.x0,
                                 ax_pos.y0 - button_height - 0.01,
                                 ax_pos.width,
                                 button_height])
        terminate_button = Button(ax_terminate, 'Terminate', color='red', hovercolor='lightcoral')
        terminate_button.on_clicked(self.terminate)

        # Create status buttons for the activities.
        num_activities = len(self.activities)
        panel_width = 0.25
        panel_height = min(0.8, 0.03 * num_activities)
        ax_radio = plt.axes([ax_pos.x0 + ax_pos.width + 0.01,
                             ax_pos.y0,
                             panel_width,
                             panel_height])
        self.radio = RadioButtons(ax_radio, self.activities, active=0)
        # Optionally adjust font size for many activities
        for text in self.radio.labels:
            text.set_fontsize(max(8, 16 - num_activities // 2))
        self.radio.on_clicked(self.update_status)

        anim = FuncAnimation(self.fig, self.update_img, interval=10,
                             cache_frame_data=False, blit=False)
        plt.show()

    def _paper_tape_loop(self):
        """
        Automate status changes based on paper tape sequence
        """
        sample_interval = getattr(self, 'visualization_interval', 0.01)
        
        # Start with an "open" period if start_delay is defined
        if self.start_delay > 0:
            self.shared_status.status = 'open'
            logging.info(f'Paper tape starting with "open" for {self.start_delay} samples.')
            time.sleep(self.start_delay * sample_interval)

        for act, dur in zip(self.tape_activities, self.tape_durations):
            if self.terminate_loop_evt.is_set():
                break
            self.shared_status.status = act
            logging.info(f'Paper tape setting status: {act}')
            time.sleep(dur * sample_interval)
            if self.terminate_loop_evt.is_set():
                break
            self.shared_status.status = 'open'
            logging.info('Paper tape setting status: open')
            time.sleep(self.tape_interval * sample_interval)
        logging.info('Completed paper tape sequence.')

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

        paper_tape_mode = bool(self.paper_tape_name and self.tape_activities)
        if not self.disable_csv:
            self.saving_process = Process(
                target=self._saving_process,
                args=(
                    self.saving_queue,
                    self.filepath,
                    self.terminate_loop_evt,
                    self.shared_status,
                    self.action_duration,
                    paper_tape_mode,
                )
            )
            self.saving_process.start()

        # start paper tape automation if configured
        if paper_tape_mode:
            # interval between frames in seconds (FuncAnimation interval=10ms)
            self.visualization_interval = 0.01
            self.tape_thread = Thread(target=self._paper_tape_loop, daemon=True)
            self.tape_thread.start()
        # main visualization process
        self._visualization_process()