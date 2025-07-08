import pyautogui
import time
from pynput import keyboard
import argparse

# Flag to control the main loop and script termination
running = True
selected_action = "region" # Default action

def perform_action():
    """Performs the selected keyboard action."""
    global selected_action
    if selected_action == "region":
        print("Performing action: T, V, E (Region)")
        pyautogui.press('t')
        time.sleep(0.05)
        pyautogui.press('v')
        time.sleep(0.05)
        pyautogui.press('e')
    elif selected_action == "board_cutout":
        print("Performing action: T, V, B (Board Cutout)")
        pyautogui.press('t')
        time.sleep(0.05)
        pyautogui.press('v')
        time.sleep(0.05)
        pyautogui.press('b')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Altium key sequences.")
    parser.add_argument(
        '--action',
        type=str,
        choices=['region', 'board_cutout'],
        default='region',
        help="Specify the action: 'region' (T-V-E) or 'board_cutout' (T-V-B). Default is 'region'."
    )
    args = parser.parse_args()
    selected_action = args.action

    print(f"Script running. Action set to: {selected_action.upper()}. Press 'Alt' key to perform action. Press 'q' to quit.")

    # Set up listener for keyboard only
    keyboard_listener = keyboard.Listener(on_press=on_press)

    keyboard_listener.start()

    # Keep the script running until 'q' is pressed
    keyboard_listener.join()

    print("Script terminated.")