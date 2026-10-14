"""
CardioPulse AI - Raspberry Pi Zero 2W & 0.96" SSD1306 OLED Companion Software

Hardware Setup on Raspberry Pi Zero 2W:
1. 0.96" I2C SSD1306 OLED Display:
   - VCC  -> 3.3V (Pin 1)
   - GND  -> GND (Pin 6)
   - SDA  -> GPIO 2 / SDA (Pin 3)
   - SCL  -> GPIO 3 / SCL (Pin 5)
2. INMP441 MEMS Digital Microphone (I2S):
   - VCC  -> 3.3V (Pin 17)
   - GND  -> GND (Pin 9)
   - SCK  -> GPIO 18 / BCLK (Pin 12)
   - WS   -> GPIO 19 / LRCLK (Pin 35)
   - SD   -> GPIO 20 / DIN (Pin 38)
   - L/R  -> GND
3. Push Button (Record Trigger):
   - Pin 1 -> GPIO 4 (Pin 7)
   - Pin 2 -> GND (Pin 14)
"""

import time
import os
import sys
import numpy as np

from PIL import Image, ImageDraw, ImageFont

# Hardware Display imports (Fallback to console UI if running on desktop test environment)
try:
    import board
    import busio
    import adafruit_ssd1306
    HAS_OLED_HARDWARE = True
except Exception:
    HAS_OLED_HARDWARE = False

import requests

SERVER_URL = "http://127.0.0.1:5000/api/predict"
SAMPLE_RATE = 16000
DURATION = 5

class StethoscopeOLEDDisplay:
    def __init__(self, width=128, height=64):
        self.width = width
        self.height = height
        self.oled = None
        
        if HAS_OLED_HARDWARE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.oled = adafruit_ssd1306.SSD1306_I2C(width, height, i2c)
                self.oled.fill(0)
                self.oled.show()
                print("[+] Hardware 0.96\" SSD1306 OLED Initialized Successfully!")
            except Exception as e:
                print(f"[!] OLED Hardware Init Warning: {e}")
                
        # Create PIL canvas buffer
        self.image = Image.new("1", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

    def show_standby_screen(self, ip_addr="127.0.0.1"):
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.draw.text((8, 2), "CardioPulse AI", font=self.font, fill=255)
        self.draw.line((0, 16, 128, 16), fill=255)
        self.draw.text((4, 22), "Stethoscope Ready", font=self.font, fill=255)
        self.draw.text((4, 36), f"IP: {ip_addr}", font=self.font, fill=255)
        self.draw.text((4, 50), "Press Button -> Rec", font=self.font, fill=255)
        self._render()

    def show_recording_screen(self, seconds_left=5):
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.draw.text((10, 4), "RECORDING HEART...", font=self.font, fill=255)
        
        # Draw animated soundwave
        for x in range(0, 128, 4):
            h = np.random.randint(5, 25)
            self.draw.line((x, 36 - h//2, x, 36 + h//2), fill=255)
            
        self.draw.text((35, 50), f"Time: 00:0{seconds_left}s", font=self.font, fill=255)
        self._render()

    def show_result_screen(self, prediction, confidence):
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.draw.text((20, 2), "AI DIAGNOSIS", font=self.font, fill=255)
        self.draw.line((0, 14, 128, 14), fill=255)
        
        status_str = prediction.upper()
        if "NORMAL" in status_str:
            title = "HEALTHY RHYTHM"
        elif "MURMUR" in status_str:
            title = "MURMUR DETECTED"
        else:
            title = "RECORD ARTIFACT"
            
        self.draw.text((4, 20), title, font=self.font, fill=255)
        self.draw.text((4, 36), f"Conf: {confidence:.1f}%", font=self.font, fill=255)
        self.draw.text((4, 50), "Press to re-record", font=self.font, fill=255)
        self._render()

    def _render(self):
        if self.oled and HAS_OLED_HARDWARE:
            self.oled.image(self.image)
            self.oled.show()
        else:
            # Console preview printout for desktop testing
            pass

def main():
    display = StethoscopeOLEDDisplay()
    print("\n==================================================")
    print("  Raspberry Pi Zero 2W Digital Stethoscope System ")
    print("==================================================")
    
    display.show_standby_screen("192.168.1.105")
    time.sleep(2)

    print("[*] Simulating Stethoscope Audio Recording (5 seconds)...")
    for sec in range(5, 0, -1):
        display.show_recording_screen(sec)
        time.sleep(1)

    print("[*] Displaying AI Diagnostic Result on OLED Screen...")
    display.show_result_screen("normal", 99.8)
    time.sleep(3)
    print("[+] Test Completed Successfully!")

if __name__ == "__main__":
    main()
