
#define NIR_LED_DUTY 10

void setup()
{
    for (int pin = 2; pin <= 10; pin++)
    {
        pinMode(pin, OUTPUT);
        analogWrite(pin, NIR_LED_DUTY);
    }

    pinMode(LED_BUILTIN, OUTPUT);
}

void loop()
{
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
}
