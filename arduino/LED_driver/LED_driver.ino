
void setup()
{
    for (int pin = 2; pin <= 10; pin++)
    {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, HIGH);
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
