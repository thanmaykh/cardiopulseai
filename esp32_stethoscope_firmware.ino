/*
 * CardioPulse AI - ESP32 Electronic Stethoscope Firmware
 * 
 * Hardware Requirements:
 * 1. ESP32 Development Board (ESP32-WROOM-32 / ESP32-S3)
 * 2. Electronic Stethoscope Module or INMP441 I2S Microphone / MAX9814 ADC
 * 3. Push Button (Record Trigger) on Pin GPIO 4
 * 4. Status LED on Pin GPIO 2
 *
 * I2S Microphone Pinout (INMP441):
 * - SCK  -> GPIO 14
 * - WS   -> GPIO 15
 * - SD   -> GPIO 32
 * - L/R  -> GND (Left Channel)
 * - VDD  -> 3.3V
 * - GND  -> GND
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>

// ================= USER CONFIGURATION =================
const char* WIFI_SSID     = "YOUR_WIFI_NAME";        // <--- Change to your Wi-Fi SSID
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";    // <--- Change to your Wi-Fi Password
const char* SERVER_URL    = "http://10.124.3.43:5000/api/esp32/upload"; // <--- Change PC IP

#define RECORD_BUTTON_PIN 4
#define STATUS_LED_PIN    2
#define SAMPLE_RATE       16000
#define RECORD_TIME_SEC   5
#define BUFFER_SIZE       (SAMPLE_RATE * RECORD_TIME_SEC * 2) // 16-bit PCM = 160,000 bytes

// I2S Configuration
#define I2S_WS            15
#define I2S_SCK           14
#define I2S_SD            32
#define I2S_PORT          I2S_NUM_0

uint8_t* pcm_buffer = NULL;

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
    Serial.begin(115200);
    pinMode(RECORD_BUTTON_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    pcm_buffer = (uint8_t*) malloc(BUFFER_SIZE);
    if (!pcm_buffer) {
        Serial.println("[!] RAM Allocation Failed!");
        while (1);
    }

    Serial.printf("[*] Connecting to Wi-Fi: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
    }
    digitalWrite(STATUS_LED_PIN, HIGH);
    Serial.println("\n[+] Wi-Fi Connected!");
    Serial.printf("[+] ESP32 IP Address: %s\n", WiFi.localIP().toString().c_str());

    setupI2S();
    Serial.println("[+] Electronic Stethoscope Ready! Press Button on GPIO 4 to Record.");
}

void writeWavHeader(uint8_t* header, uint32_t pcmDataSize) {
    uint32_t fileSize = pcmDataSize + 36;
    uint32_t sampleRate = SAMPLE_RATE;
    uint32_t byteRate = SAMPLE_RATE * 1 * 2;

    memcpy(header, "RIFF", 4);
    memcpy(header + 4, &fileSize, 4);
    memcpy(header + 8, "WAVEfmt ", 8);
    
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels = 1; // Mono
    uint16_t bitsPerSample = 16;
    uint16_t blockAlign = 2;

    memcpy(header + 16, &subchunk1Size, 4);
    memcpy(header + 20, &audioFormat, 2);
    memcpy(header + 22, &numChannels, 2);
    memcpy(header + 24, &sampleRate, 4);
    memcpy(header + 28, &byteRate, 4);
    memcpy(header + 32, &blockAlign, 2);
    memcpy(header + 34, &bitsPerSample, 2);
    memcpy(header + 36, "data", 4);
    memcpy(header + 40, &pcmDataSize, 4);
}

void sendAudioToServer() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[!] Wi-Fi Disconnected!");
        return;
    }

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "audio/wav");

    uint8_t wavHeader[44];
    writeWavHeader(wavHeader, BUFFER_SIZE);

    uint8_t* fullWavPayload = (uint8_t*) malloc(44 + BUFFER_SIZE);
    memcpy(fullWavPayload, wavHeader, 44);
    memcpy(fullWavPayload + 44, pcm_buffer, BUFFER_SIZE);

    Serial.println("[*] Transmitting 5s Stethoscope Recording to CardioPulse AI...");
    int httpResponseCode = http.POST(fullWavPayload, 44 + BUFFER_SIZE);

    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.printf("[+] Server Response (%d): %s\n", httpResponseCode, response.c_str());
        
        if (response.indexOf("normal") > 0) {
            digitalWrite(STATUS_LED_PIN, LOW); delay(200);
            digitalWrite(STATUS_LED_PIN, HIGH); delay(500);
        } else if (response.indexOf("murmur") > 0) {
            for (int i = 0; i < 3; i++) {
                digitalWrite(STATUS_LED_PIN, LOW); delay(150);
                digitalWrite(STATUS_LED_PIN, HIGH); delay(150);
            }
        }
    } else {
        Serial.printf("[!] HTTP POST Error: %s\n", http.errorToString(httpResponseCode).c_str());
    }

    free(fullWavPayload);
    http.end();
}

void loop() {
    if (digitalRead(RECORD_BUTTON_PIN) == LOW) {
        delay(50);
        if (digitalRead(RECORD_BUTTON_PIN) == LOW) {
            Serial.println("\n[*] Recording Stethoscope Sound (5 seconds)...");
            digitalWrite(STATUS_LED_PIN, LOW);

            size_t bytesRead = 0;
            size_t totalBytesRead = 0;

            while (totalBytesRead < BUFFER_SIZE) {
                i2s_read(I2S_PORT, pcm_buffer + totalBytesRead, 1024, &bytesRead, portMAX_DELAY);
                totalBytesRead += bytesRead;
            }

            digitalWrite(STATUS_LED_PIN, HIGH);
            Serial.println("[+] Recording Complete! Sending over Wi-Fi...");
            sendAudioToServer();
            
            delay(1000);
        }
    }
    delay(10);
}
