import pyautogui
import time
from pynput import keyboard

# Flag to control the main loop and script termination
running = True

def perform_action():
    """Performs the selected keyboard action."""
    print("Performing action: Tab -> A -> P -> L -> Tab")
    pyautogui.press('tab')
    time.sleep(0.05)
    pyautogui.press('a')
    time.sleep(0.05)
    pyautogui.press('p')
    time.sleep(0.05)
    pyautogui.press('l')
    time.sleep(0.05)
    pyautogui.press('tab')
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
        pass

def on_release(key):
    """Handles keyboard release events."""
    if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
        print("'Alt' key released.")
        perform_action()

if __name__ == "__main__":
    print("Script running. Release 'Alt' key to perform action. Press 'q' to quit.")

    # Set up listener for keyboard press and release
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    keyboard_listener.start()

    # Keep the script running until 'q' is pressed
    keyboard_listener.join()

    print("Script terminated.")
