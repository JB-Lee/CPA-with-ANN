from struct import unpack
from typing import Union, List

import numpy as np
import pyvisa as visa


class ScopeManager:
    resource_manager: visa.ResourceManager
    scope: visa.Resource

    def __init__(self):
        self.visa_init()
        self.scope_init()

    def visa_init(self):
        self.resource_manager = visa.ResourceManager()
        instruments = self.resource_manager.list_resources()
        self.scope = self.resource_manager.open_resource(instruments[0])

    def scope_init(self):
        self.scope.write('DATA:SOU CH1')
        self.scope.write('DATA:WIDTH 1')
        self.scope.write('DATA:ENC RP8')

    def get_data(self, channels: Union[List[str], str]):
        def _get_data(channel):
            self.scope.write(f'DATA:SOU {channel}')

            y_mult = float(self.scope.query('WFMPRE:YMULT?'))
            y_zero = float(self.scope.query('WFMPRE:YZERO?'))
            y_off = float(self.scope.query('WFMPRE:YOFF?'))
            x_incr = float(self.scope.query('WFMPRE:XINCR?'))

            self.scope.write('CURVE?')

            data = self.scope.read_raw()
            header_len = 2 + int(data[1])
            header = data[:header_len]
            ADC_wave = data[header_len:-1]

            ADC_wave = np.array(unpack('%sB' % len(ADC_wave), ADC_wave))

            Volts = (ADC_wave - y_off) * y_mult + y_zero

            Time = np.arange(0, x_incr * len(Volts), x_incr)

            return Volts, Time

        if isinstance(channels, str):
            return _get_data(channels)

        result = [_get_data(channel) for channel in channels]

        return result
