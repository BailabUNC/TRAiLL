import pyautogui
import time
from pynput import keyboard
import argparse

# Flag to control the main loop
running = True
# Default target coordinates (e.g., for 4K screen)
DEFAULT_TARGET_X, DEFAULT_TARGET_Y = 3790, 444
# Target coordinates for 2K screen
TARGET_2K_X, TARGET_2K_Y = 2526, 449

# Global variables for target coordinates, to be set based on args
TARGET_X, TARGET_Y = DEFAULT_TARGET_X, DEFAULT_TARGET_Y

def perform_click_action():
    """Moves to target, clicks, and returns to original position."""
    try:
        original_x, original_y = pyautogui.position()
        print(f"Original cursor position: ({original_x}, {original_y})")
        
        pyautogui.moveTo(TARGET_X, TARGET_Y)
        print(f"Moved cursor to: ({TARGET_X}, {TARGET_Y})")
        
        pyautogui.click() # Performs a left click by default
        print("Performed left click.")
        
        # Short delay to ensure click registers before moving back
        time.sleep(0.01)
        
        pyautogui.moveTo(original_x, original_y)
        print(f"Returned cursor to: ({original_x}, {original_y})")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def on_press(key):
    global running
    try:
        if key.char == 'q':
            print("\n'q' pressed, stopping script.")
            running = False
            return False  # Stop the listener
    except AttributeError:
        # Special keys (like alt, shift, ctrl, etc.)
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            print("\nAlt key pressed.")
            perform_click_action()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate mouse clicks for different screen resolutions.")
    parser.add_argument('--screen', type=str, default='4k', choices=['4k', '2k'],
                        help="Specify screen resolution: '4k' (default) or '2k'.")
    args = parser.parse_args()

    if args.screen == '2k':
        TARGET_X, TARGET_Y = TARGET_2K_X, TARGET_2K_Y
        print("Using 2K screen target coordinates.")
    else:
        # TARGET_X, TARGET_Y are already set to default (4k)
        print("Using 4K screen target coordinates (default).")


    print(f"Script running. Press 'Alt' to move to ({TARGET_X}, {TARGET_Y}), click, and return. Press 'q' to quit.")

    # Set up keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Keep the script running until 'q' is pressed or listener stops
    listener.join()

    print("Script terminated.")
