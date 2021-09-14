from struct import pack

import serial

PROTO_END_OF_TRANSFER = b'\x0a'
PROTO_CMD_INITIALIZE = b'\x01'
PROTO_CMD_ENCRYPT = b'\x02'
PROTO_CMD_FINALIZE = b'\x03'


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


class ArduinoProtocol(BaseArduino):
    def __check_connection(self):
        if not self.is_connected:
            raise Exception('ConnectionError')

    def initialize(self, plain_text, key):
        self.__check_connection()

        self.serial.write(
            pack('c16s16sc', PROTO_CMD_INITIALIZE, plain_text, key, PROTO_END_OF_TRANSFER)
        )

        return self.serial.read(1)

    def encrypt(self):
        self.__check_connection()

        self.serial.write(
            pack('cc', PROTO_CMD_ENCRYPT, PROTO_END_OF_TRANSFER)
        )

    def finalize(self):
        self.__check_connection()

        self.serial.write(
            pack('cc', PROTO_CMD_FINALIZE, PROTO_END_OF_TRANSFER)
        )

        result = self.serial.read(16)
        return result

    def wait_encrypt_end(self):
        return self.serial.read(1)
