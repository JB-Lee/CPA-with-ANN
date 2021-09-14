import serial


class BaseArduino:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_connected = False
        self.serial = None

    def connect(self):
        if not self.is_connected:
            self.serial = serial.Serial(*self.args, **self.kwargs)
            self.is_connected = True
            self.serial.read(1)
            self.serial.read(1)

    def close(self):
        if self.is_connected:
            self.is_connected = False
            self.serial.close()
