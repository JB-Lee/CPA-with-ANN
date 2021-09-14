#include "serial_helper.h"

char PLAIN_TEXT[16]{};
char KEY[16]{};
char ENCRYPTED[16]{};

const unsigned long BAUDRATE = 9600;
const unsigned long TIMEOUT = 5;

const byte PROTOCOL_CMD_INITIALIZE = 0x01;
const byte PROTOCOL_CMD_ENCRYPT = 0x02;
const byte PROTOCOL_CMD_FINALIZE = 0x03;

const byte PROTOCOL_END_INITIALIZE = 0x11;
const byte PROTOCOL_END_ENCRYPT = 0x12;

SerialHelper sh(BAUDRATE, TIMEOUT);

void Encrypt()
{
    digitalWrite(13, HIGH);
    for (int i = 0; i < 16; ++i)
    {
        ENCRYPTED[i] = PLAIN_TEXT[i] ^ KEY[i];
    }
    digitalWrite(13, LOW);
}

// ####################################################
// ## Listeners
// ####################################################

void onInitialize()
{
    //Plain Text initialize
    Serial.readBytes(PLAIN_TEXT, 16);

    //Key initialize
    Serial.readBytes(KEY, 16);

    Serial.write(PROTOCOL_END_INITIALIZE);
}

void onEncrypt()
{
    Encrypt();
    Serial.write(PROTOCOL_END_ENCRYPT);
}

void onFinalize()
{
    Serial.write(ENCRYPTED, 16);
}

// ####################################################
// ## setup, loop
// ####################################################

void setup()
{
    pinMode(13, OUTPUT);
    digitalWrite(13, LOW);

    sh.init();

    sh.register_command(PROTOCOL_CMD_INITIALIZE, onInitialize);
    sh.register_command(PROTOCOL_CMD_ENCRYPT, onEncrypt);
    sh.register_command(PROTOCOL_CMD_FINALIZE, onFinalize);

    Serial.flush();
}

void loop()
{
    sh.loop();
}
