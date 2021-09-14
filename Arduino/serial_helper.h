const byte END_OF_TRANSFER = 0x0A;

class SerialHelper {

public:
    SerialHelper(
        unsigned long baudrate,
        unsigned long timeout
    );

    typedef void (*CallbackPointer)();
    unsigned long baudrate;
    unsigned long timeout;

    void register_command(byte command, CallbackPointer ptr);
    void loop();
    void init();


private:
    CallbackPointer callbackpointers[256];
    void flushUntil(byte terminator);
};