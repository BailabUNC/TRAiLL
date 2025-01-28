
#define NUM_MUX         3   // Number of multiplexers
#define NUM_CHANNEL     16  // Number of channels in a mux
#define MAX_NUM_READING 36

#define LOCAL_ADC_RES   12
#define MAX_ADC_VALUE   (1 << LOCAL_ADC_RES) - 1  // for 12-bit, 4095

static const uint8_t pin_mapping[3][4] = {
    {2,  3,  6,  7 },   // Group 0
    {8,  9,  10, 11},   // Group 1
    {12, 13, 14, 15}    // Group 2
};

void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    while(!Serial);
    Serial.println("begin");

    analogReadResolution(LOCAL_ADC_RES);

    for (int group = 0; group < NUM_MUX; group++)
    {
        for (int index = 0; index < 4; index++)
            pinMode(pin_mapping[group][index], OUTPUT);
    }

    pinMode(LED_BUILTIN, OUTPUT);

}

void setMuxChannel(int group, int channel)
{
    // Extract the bits from channel
    int n1 = (channel >> 3) & 1;    // MSB (bit 3)
    int n2 = (channel >> 2) & 1;    // Bit 2
    int n3 = (channel >> 1) & 1;    // Bit 1
    int r  =  channel       & 1;    // LSB (bit 0)

    // Write bits to pins
    digitalWrite(pin_mapping[group][0], r);
    digitalWrite(pin_mapping[group][1], n3);
    digitalWrite(pin_mapping[group][2], n2);
    digitalWrite(pin_mapping[group][3], n1);
}

int getChannelRead(int channel)
{
    int read_val;

    if (channel < 16)
    {
        setMuxChannel(0, channel);
        read_val = analogRead(A0);
    }
    else if (channel < 32)
    {
        setMuxChannel(1, channel - 16);
        read_val = analogRead(A1);
    }
    else if (channel < 48)
    {
        setMuxChannel(2, channel - 32);
        read_val = analogRead(A2);
    }
    else
        return -1;

    return read_val;
}

void loop()
{
    digitalWrite(LED_BUILTIN, HIGH);

    for (int row = 0; row < 6; row++)
    {
        for (int col = 0; col < 6; col++)
        {
            int channel = row * 6 + col;
            int val = getChannelRead(channel);

            Serial.print(val);
            Serial.print('\t');
        }

        Serial.println();
    }

    Serial.println();

    digitalWrite(LED_BUILTIN, LOW);
    delay(10);
}






