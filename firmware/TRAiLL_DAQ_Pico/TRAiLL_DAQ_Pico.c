#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/adc.h"
#include "hardware/timer.h"

// Define constants for multiplexer configuration
#define NUM_MUX         3   // Number of multiplexers
#define NUM_CHANNEL     16  // Number of channels per multiplexer
#define MAX_NUM_READING 36

#define LOCAL_ADC_RES   12
#define MAX_ADC_VALUE   ((1 << LOCAL_ADC_RES) - 1)  // 4095 for 12-bit ADC

#define ALARM_US        20000

// For example, group 0 uses pins 2, 3, 6, 7, etc.
static const uint8_t pin_mapping[NUM_MUX][4] = {
    {2,  3,  6,  7},     // Group 0
    {8,  9,  10, 11},    // Group 1
    {12, 13, 14, 15}     // Group 2
};

// Set the multiplexer channel for a given group by writing out its 4 control bits.
// The mapping is: pin0 gets the LSB, then pin1 gets bit 1 (n3), pin2 gets bit 2 (n2),
// and pin3 gets the MSB (n1).
void setMuxChannel(int group, int channel)
{
    int n1 = (channel >> 3) & 1;
    int n2 = (channel >> 2) & 1;
    int n3 = (channel >> 1) & 1;
    int r  =  channel       & 1;

    gpio_put(pin_mapping[group][0], r);
    gpio_put(pin_mapping[group][1], n3);
    gpio_put(pin_mapping[group][2], n2);
    gpio_put(pin_mapping[group][3], n1);
}

// Read an analog channel based on the overall channel number.
// Channels 0-15 use multiplexer group 0 (ADC0 on GPIO26),
// channels 16-31 use group 1 (ADC1 on GPIO27),
// channels 32-47 use group 2 (ADC2 on GPIO28).
uint16_t getChannelRead(int channel)
{
    uint16_t read_val;
    if (channel < 16)
    {
        setMuxChannel(0, channel);
        adc_select_input(0);  // ADC0 (GPIO26)
        read_val = adc_read();
    }
    else if (channel < 32)
    {
        setMuxChannel(1, channel - 16);
        adc_select_input(1);  // ADC1 (GPIO27)
        read_val = adc_read();
    }
    else if (channel < 48)
    {
        setMuxChannel(2, channel - 32);
        adc_select_input(2);  // ADC2 (GPIO28)
        read_val = adc_read();
    }
    else
    {
        read_val = 0; // Invalid channel
    }
    return read_val;
}

int64_t alarm_callback(alarm_id_t id, void *user_data)
{
    gpio_put(PICO_DEFAULT_LED_PIN, 1);

    // Loop over a 6x6 grid (36 channels)
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 6; col++) {
            int channel = row * 6 + col;
            uint16_t val = getChannelRead(channel);
            printf("%u\t", val);
        }
        printf("\n");
    }
    printf("\n");

    gpio_put(PICO_DEFAULT_LED_PIN, 0);

    // Return the number of microseconds until the next callback
    return ALARM_US;
}

int main()
{
    stdio_init_all();

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    adc_init();
    adc_gpio_init(26);
    adc_gpio_init(27);
    adc_gpio_init(28);

    for (int group = 0; group < NUM_MUX; group++)
    {
        for (int i = 0; i < 4; i++)
        {
            gpio_init(pin_mapping[group][i]);
            gpio_set_dir(pin_mapping[group][i], GPIO_OUT);
        }
    }

    // Optionally, wait a moment for stdio (e.g., USB serial) to initialize.
    // sleep_ms(2000);

    // Set up a HW timer that fires in 10ms
    add_alarm_in_ms(10, alarm_callback, NULL, true);

    while (true)
    {
        tight_loop_contents();
    }

    return 0;
}
