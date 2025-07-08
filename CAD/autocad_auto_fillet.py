import time
import pyautogui
from pynput import keyboard

# pynput keyboard controller is no longer needed for pressing enter

def on_press(key):
    try:
        if key.char == 'f':
            print("'f' pressed. Simulating 'enter' in 1 second...")
            time.sleep(1)
            pyautogui.press('enter') # Use pyautogui to press enter
            print("'enter' pressed.")
        elif key.char == 'q':
            print("Exiting script...")
            # Stop listener
            return False
    except AttributeError:
        # Special keys (like shift, ctrl, etc.) don't have a char attribute
        pass

# Collect events until released
print("Script started. Press 'f' to trigger 'enter' after 1s. Press 'q' to quit.")
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

print("Script stopped.")