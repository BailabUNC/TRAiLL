import pyautogui
import time
from pynput import keyboard

# Flag to control the main loop and script termination
running = True

def perform_action():
    """Performs the T, V, E keyboard action."""
    print("Performing action: T, V, E")
    pyautogui.press('t')
    time.sleep(0.05)
    pyautogui.press('v')
    time.sleep(0.05)
    pyautogui.press('e')
    print("Action finished.")

def on_press(key):
    """Handles keyboard press events."""
    global running
    try:
        if key.char == 'q':
            print("'q' pressed, stopping script.")
            running = False
            # Stop listener
            keyboard_listener.stop()
            return False # Stop the keyboard listener
    except AttributeError:
        # Special keys (like shift, ctrl, alt, etc.) don't have a char attribute
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            print("'Alt' key pressed.")
            perform_action()
        pass

print("Script running. Press 'Alt' key to perform action. Press 'q' to quit.")

# Set up listener for keyboard only
keyboard_listener = keyboard.Listener(on_press=on_press)

keyboard_listener.start()

# Keep the script running until 'q' is pressed
keyboard_listener.join()

print("Script terminated.")