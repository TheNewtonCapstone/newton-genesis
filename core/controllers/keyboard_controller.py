from pynput import keyboard
import torch

class KeyboardController:
    def __init__(self, command_scale=0.5):
        """
        Keyboard controller for manual movement input.

        Args:
            command_scale (float): Scaling factor for velocity commands.
        """
        self.command_scale = command_scale
        self.current_command = torch.zeros(3)  # [lin_x, lin_y, ang_z]
        self.key_state = {  # Track pressed keys
            keyboard.Key.up: False,
            keyboard.Key.down: False,
            keyboard.Key.left: False,
            keyboard.Key.right: False
        }
        self.active = False  # Control mode toggle
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.esc:  # Toggle manual control
            self.active = not self.active
            print(f"Manual control {'enabled' if self.active else 'disabled'}.")
            return

        if key in self.key_state:
            self.key_state[key] = True
        self.update_command()

    def on_release(self, key):
        if key in self.key_state:
            self.key_state[key] = False
        self.update_command()

    def update_command(self):
        """ Update movement commands based on pressed keys. Reset if none are pressed. """
        if not any(self.key_state.values()):
            self.current_command = torch.zeros(3)  # Reset to zero when no keys are pressed
        else:
            self.current_command = torch.zeros(3)
            if self.key_state[keyboard.Key.up]:
                self.current_command[0] += self.command_scale  # Forward
            if self.key_state[keyboard.Key.down]:
                self.current_command[0] -= self.command_scale  # Backward
            if self.key_state[keyboard.Key.left]:
                self.current_command[2] += self.command_scale  # Rotate left
            if self.key_state[keyboard.Key.right]:
                self.current_command[2] -= self.command_scale  # Rotate right

    def get_command(self):
        """ Retrieve the current velocity command. """
        return self.current_command if self.active else None
