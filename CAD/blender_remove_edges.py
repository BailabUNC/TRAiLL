import pyautogui
import time
from pynput import keyboard

# Flag to control the main loop and script termination
running = True

def perform_action():
    """Performs the selected keyboard action."""
    print("Performing action: Tab -> (LMB drag + X + E) * 4 -> Tab")
    pyautogui.press('tab')
    time.sleep(0.1)
    for i in range(4):
        print(f"Iteration {i+1}: LMB drag, X, E")
        # Press and hold LMB
        pyautogui.mouseDown()
        time.sleep(0.02)
        # Move mouse 250 pixels to the right
        pyautogui.moveRel(250, 5, duration=0.15)
        time.sleep(0.02)
        # Release LMB
        pyautogui.mouseUp()
        time.sleep(0.02)
        # Press X
        pyautogui.press('x')
        time.sleep(0.02)
        # Press E
        pyautogui.press('e')
        time.sleep(0.02)
        # Move mouse 250 pixels to the left
        pyautogui.moveRel(-250, -5, duration=0.15)
        time.sleep(0.02)
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
