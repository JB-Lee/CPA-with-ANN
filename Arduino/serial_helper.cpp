#include "Arduino.h"
#include "serial_helper.h"

//Public

SerialHelper::SerialHelper(
    unsigned long baudrate,
    unsigned long timeout
) : baudrate(baudrate), timeout(timeout), callbackpointers() {
}

void SerialHelper::register_command(byte command, CallbackPointer ptr) {
    this->callbackpointers[command] = ptr;
}

void SerialHelper::loop() {
    while(!Serial.available());

    byte cmd = Serial.read();

    if (!callbackpointers[cmd]) {
        this->flushUntil(END_OF_TRANSFER);
        return;
    }

    this->callbackpointers[cmd]();
    this->flushUntil(END_OF_TRANSFER);
}

void SerialHelper::init() {
    Serial.begin(this->baudrate);
    Serial.setTimeout(this->timeout);
}

//Private

void SerialHelper::flushUntil(byte terminator) {
    while(Serial.available()) {
        if (Serial.read() == terminator) return;
    }
}