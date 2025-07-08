#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/adc.h"
#include "hardware/timer.h"
#include "hardware/pwm.h"

// Define constants for multiplexer configuration
#define NUM_MUX         3   // Number of multiplexers
#define NUM_CHANNEL     16  // Number of channels per multiplexer
#define MAX_ROW         6
#define MAX_COL         8
#define MAX_NUM_READING 48

#define LOCAL_ADC_RES   12
#define MAX_ADC_VALUE   ((1 << LOCAL_ADC_RES) - 1)  // 4095 for 12-bit ADC

#define LED_CTRL_PIN    22
#define PWM_WRAP_VALUE  (MAX_ADC_VALUE / 2)

static const uint8_t pin_mapping[NUM_MUX][4] = {
    {2,  3,  6,  7},     // Group 0
    {8,  9,  10, 11},    // Group 1
    {12, 13, 14, 15}     // Group 2
};

// Set the multiplexer channel for a given group by writing out its 4 control bits.
void setMuxChannel(int group, int channel)
{
    int s3 = (channel >> 3) & 1;
    int s2 = (channel >> 2) & 1;
    int s1 = (channel >> 1) & 1;
    int s0 =  channel       & 1;

    gpio_put(pin_mapping[group][0], s0);
    gpio_put(pin_mapping[group][1], s1);
    gpio_put(pin_mapping[group][2], s2);
    gpio_put(pin_mapping[group][3], s3);
}

void pwmInit(uint gpio_pin)
{
    gpio_set_function(gpio_pin, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(gpio_pin);

    pwm_config config = pwm_get_default_config();
    pwm_config_set_wrap(&config, MAX_ADC_VALUE);
    
    pwm_init(slice_num, &config, true);
}

// Read an analog channel based on the overall channel number.
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

// Repeating timer callback: This function will be called every 1000ms.
bool repeating_timer_callback(struct repeating_timer *t)
{
    // Turn on LEDs before reading
    gpio_put(PICO_DEFAULT_LED_PIN, 1);
    pwm_set_gpio_level(LED_CTRL_PIN, (MAX_ADC_VALUE / 2));
    
    // Loop over a 6x8 grid (48 channels)
    for (int row = 0; row < MAX_ROW; row++)
    {
        for (int col = 0; col < MAX_COL; col++)
        {
            int channel = row * MAX_COL + col;
            uint16_t val = getChannelRead(channel);
            printf("%u\t", val);
        }
        printf("\n");
    }
    printf("\n");
    
    // Turn off LEDs after reading
    pwm_set_gpio_level(LED_CTRL_PIN, 0);
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
    
    return true; // Keep the timer running
}
    
int main()
{
    stdio_init_all();

    // Initialize LED control pins
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    // Set up PWM for LED control
    pwmInit(LED_CTRL_PIN);

    // Initialize ADC and its GPIOs
    adc_init();
    adc_gpio_init(26);
    adc_gpio_init(27);
    adc_gpio_init(28);

    // Initialize multiplexer control pins
    for (int group = 0; group < NUM_MUX; group++)
    {
        for (int i = 0; i < 4; i++)
        {
            gpio_init(pin_mapping[group][i]);
            gpio_set_dir(pin_mapping[group][i], GPIO_OUT);
        }
    }

    // Set up a repeating timer that fires every 1000ms (1 second)
    struct repeating_timer timer;
    add_repeating_timer_ms(10, repeating_timer_callback, NULL, &timer);

    // Main loop can remain idle while the timer handles the periodic tasks
    while (true)
    {
        tight_loop_contents();
    }

    return 0;
}
