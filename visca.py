import serial
import time
import threading
from typing import Optional, Tuple, Union


class VISCAError(Exception):
    """Custom exception for VISCA protocol errors."""
    pass


class VISCACamera:
    """
    A basic VISCA camera control library using pyserial.

    This library provides control over PTZ cameras that support the VISCA protocol.
    """

    # VISCA command constants
    COMMAND_HEADER = 0x81
    COMMAND_TERMINATOR = 0xFF

    # Response codes
    ACK = 0x41
    COMPLETION = 0x51
    ERROR_LENGTH = 0x61
    ERROR_SYNTAX = 0x62
    ERROR_BUFFER_FULL = 0x63
    ERROR_CANCELLED = 0x64
    ERROR_NO_SOCKET = 0x65
    ERROR_NOT_EXECUTABLE = 0x66

    def __init__(self, port: str, baudrate: int = 9600, camera_address: int = 1):
        """
        Initialize VISCA camera connection.

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (default 9600)
            camera_address: VISCA address of camera (1-7, default 1)
        """
        self.port = port
        self.baudrate = baudrate
        self.camera_address = camera_address
        self.serial_connection: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """
        Establish serial connection to camera.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )

            # Clear any existing data in buffers
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()

            # Test connection with address set command
            return self._address_set()

        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

    def _send_command(self, command: list) -> bytes:
        """
        Send VISCA command and wait for response.

        Args:
            command: List of command bytes (without header and terminator)

        Returns:
            Response bytes from camera
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            raise VISCAError("Not connected to camera")

        with self._lock:
            # Build complete command
            full_command = [0x80 | self.camera_address] + command + [self.COMMAND_TERMINATOR]
            command_bytes = bytes(full_command)

            # Send command
            self.serial_connection.write(command_bytes)

            # Read response
            response = b''
            start_time = time.time()

            while time.time() - start_time < 2.0:  # 2 second timeout
                if self.serial_connection.in_waiting > 0:
                    byte = self.serial_connection.read(1)
                    response += byte

                    # Check if we got complete response (ends with 0xFF)
                    if byte == bytes([self.COMMAND_TERMINATOR]):
                        break

                time.sleep(0.01)

            if not response:
                raise VISCAError("No response from camera")

            # Check for errors
            self._check_response_error(response)

            return response

    def _check_response_error(self, response: bytes):
        """Check response for error codes and raise appropriate exceptions."""
        if len(response) < 2:
            return

        response_type = response[1]

        error_messages = {
            self.ERROR_LENGTH: "Message length error",
            self.ERROR_SYNTAX: "Syntax error",
            self.ERROR_BUFFER_FULL: "Command buffer full",
            self.ERROR_CANCELLED: "Command cancelled",
            self.ERROR_NO_SOCKET: "No socket available",
            self.ERROR_NOT_EXECUTABLE: "Command not executable"
        }

        if response_type in error_messages:
            raise VISCAError(error_messages[response_type])

    def _address_set(self) -> bool:
        """Send address set command to establish communication."""
        try:
            response = self._send_command([0x30, 0x01])
            return len(response) > 0
        except VISCAError:
            return False

    def power_on(self):
        """Turn camera power on."""
        self._send_command([0x01, 0x04, 0x00, 0x02])

    def power_off(self):
        """Turn camera power off."""
        self._send_command([0x01, 0x04, 0x00, 0x03])

    def home(self):
        """Move camera to home position."""
        self._send_command([0x01, 0x06, 0x04])

    def reset(self):
        """Reset camera to default settings."""
        self._send_command([0x01, 0x06, 0x05])

    def pan_tilt_up(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera up."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x03, 0x01)

    def pan_tilt_down(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera down."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x03, 0x02)

    def pan_tilt_left(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera left."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x01, 0x03)

    def pan_tilt_right(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera right."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x02, 0x03)

    def pan_tilt_up_left(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera up and left."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x01, 0x01)

    def pan_tilt_up_right(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera up and right."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x02, 0x01)

    def pan_tilt_down_left(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera down and left."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x01, 0x02)

    def pan_tilt_down_right(self, pan_speed: int = 5, tilt_speed: int = 5):
        """Move camera down and right."""
        self._pan_tilt_command(pan_speed, tilt_speed, 0x02, 0x02)

    def pan_tilt_stop(self):
        """Stop pan/tilt movement."""
        self._pan_tilt_command(0, 0, 0x03, 0x03)

    def _pan_tilt_command(self, pan_speed: int, tilt_speed: int, pan_direction: int, tilt_direction: int):
        """Send pan/tilt command with specified speeds and directions."""
        pan_speed = max(0x01, min(0x18, pan_speed))  # Clamp to valid range
        tilt_speed = max(0x01, min(0x14, tilt_speed))  # Clamp to valid range

        self._send_command([
            0x01, 0x06, 0x01,
            pan_speed, tilt_speed,
            pan_direction, tilt_direction
        ])

    def zoom_in(self, speed: int = 2):
        """Zoom in at specified speed (0-7)."""
        speed = max(0, min(7, speed))
        self._send_command([0x01, 0x04, 0x07, 0x20 | speed])

    def zoom_out(self, speed: int = 2):
        """Zoom out at specified speed (0-7)."""
        speed = max(0, min(7, speed))
        self._send_command([0x01, 0x04, 0x07, 0x30 | speed])

    def zoom_stop(self):
        """Stop zoom movement."""
        self._send_command([0x01, 0x04, 0x07, 0x00])

    def focus_near(self, speed: int = 2):
        """Focus near at specified speed (0-7)."""
        speed = max(0, min(7, speed))
        self._send_command([0x01, 0x04, 0x08, 0x30 | speed])

    def focus_far(self, speed: int = 2):
        """Focus far at specified speed (0-7)."""
        speed = max(0, min(7, speed))
        self._send_command([0x01, 0x04, 0x08, 0x20 | speed])

    def focus_stop(self):
        """Stop focus movement."""
        self._send_command([0x01, 0x04, 0x08, 0x00])

    def focus_auto(self):
        """Set focus to automatic mode."""
        self._send_command([0x01, 0x04, 0x38, 0x02])

    def focus_manual(self):
        """Set focus to manual mode."""
        self._send_command([0x01, 0x04, 0x38, 0x03])

    def get_pan_tilt_position(self) -> Tuple[int, int]:
        """
        Get current pan and tilt position.

        Returns:
            Tuple of (pan_position, tilt_position)
        """
        response = self._send_command([0x09, 0x06, 0x12])

        if len(response) >= 11:
            # Parse position from response
            pan_pos = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
            tilt_pos = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]

            # Convert from unsigned to signed
            if pan_pos > 0x8000:
                pan_pos -= 0x10000
            if tilt_pos > 0x8000:
                tilt_pos -= 0x10000

            return (pan_pos, tilt_pos)

        raise VISCAError("Invalid position response")

    def set_pan_tilt_position(self, pan_position: int, tilt_position: int, pan_speed: int = 5, tilt_speed: int = 5):
        """
        Move to absolute pan/tilt position.

        Args:
            pan_position: Target pan position
            tilt_position: Target tilt position
            pan_speed: Pan movement speed (1-24)
            tilt_speed: Tilt movement speed (1-20)
        """
        pan_speed = max(0x01, min(0x18, pan_speed))
        tilt_speed = max(0x01, min(0x14, tilt_speed))

        # Convert to unsigned values for transmission
        if pan_position < 0:
            pan_position += 0x10000
        if tilt_position < 0:
            tilt_position += 0x10000

        # Split positions into nibbles
        pan_bytes = [
            (pan_position >> 12) & 0x0F,
            (pan_position >> 8) & 0x0F,
            (pan_position >> 4) & 0x0F,
            pan_position & 0x0F
        ]

        tilt_bytes = [
            (tilt_position >> 12) & 0x0F,
            (tilt_position >> 8) & 0x0F,
            (tilt_position >> 4) & 0x0F,
            tilt_position & 0x0F
        ]

        command = [0x01, 0x06, 0x02, pan_speed, tilt_speed] + pan_bytes + tilt_bytes
        self._send_command(command)


# Example usage
if __name__ == "__main__":
    # Example usage of the VISCA camera library
    camera = VISCACamera(port='COM3', baudrate=9600, camera_address=1)  # Adjust port as needed

    try:
        if camera.connect():
            print("Connected to camera successfully!")

            # Basic camera operations
            camera.home()  # Move to home position
            time.sleep(2)

            # Pan and tilt movements
            camera.pan_tilt_right(speed=10)
            time.sleep(1)
            camera.pan_tilt_stop()

            # Zoom operations
            camera.zoom_in(speed=3)
            time.sleep(2)
            camera.zoom_stop()

            # Get current position
            pan, tilt = camera.get_pan_tilt_position()
            print(f"Current position: Pan={pan}, Tilt={tilt}")

        else:
            print("Failed to connect to camera")

    except VISCAError as e:
        print(f"VISCA Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.disconnect()
        print("Disconnected from camera")


def shake(vc, speed, pause, cnt):
    for i in range(cnt):
        vc.pan_tilt_left(speed)
        time.sleep(pause)
        vc.pan_tilt_right(speed)
        time.sleep(pause)

    vc.pan_tilt_stop()


def nod(vc, speed, pause, cnt):
    for i in range(cnt):
        vc.pan_tilt_up(speed)
        time.sleep(pause)
        vc.pan_tilt_down(speed)
        time.sleep(pause)
    vc.pan_tilt_stop()

#vc = visca.VISCACamera(v_port, camera_address=1)
#vc.connect()
shake(vc, 12, 0.2, 3)
nod(vc, 12, 0.2, 3)
