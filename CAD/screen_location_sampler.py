import pyautogui
from pynput import keyboard

# Flag to control the main loop
running = True
recorded_locations = []

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
            current_mouse_x, current_mouse_y = pyautogui.position()
            location_info = f"Alt pressed: Cursor at ({current_mouse_x}, {current_mouse_y})"
            print(location_info)
            recorded_locations.append((current_mouse_x, current_mouse_y))

print("Script running. Press 'Alt' to record cursor location. Press 'q' to quit.")

# Set up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Keep the script running until 'q' is pressed or listener stops
listener.join()

if recorded_locations:
    print("\nRecorded locations:")
    for loc in recorded_locations:
        print(loc)
else:
    print("\nNo locations were recorded.")

print("Script terminated.")
