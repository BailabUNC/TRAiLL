import pyautogui
import time
from pynput import keyboard

# Flag to control the main loop
running = True
TARGET_X, TARGET_Y = 3790, 444

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

print(f"Script running. Press 'Alt' to move to ({TARGET_X}, {TARGET_Y}), click, and return. Press 'q' to quit.")

# Set up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Keep the script running until 'q' is pressed or listener stops
listener.join()

print("Script terminated.")
