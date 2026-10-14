#!/bin/bash
# =====================================================================
# CardioPulse AI - Raspberry Pi Zero 2W One-Command Installer
# Target OS: Raspberry Pi OS (Bullseye / Bookworm 32-bit/64-bit)
# Hardware: Raspberry Pi Zero 2W + INMP441 MEMS Mic + 0.96" SSD1306 OLED
# =====================================================================

echo "=========================================================="
echo "   CardioPulse AI - Smart Stethoscope Raspberry Pi Setup  "
echo "=========================================================="

# 1. Update APT Package Lists & Install System Audio/I2C Dependencies
echo "[*] Updating System Packages & Audio Libraries..."
sudo apt update -y
sudo apt install -y python3-pip python3-dev python3-pil libasound2-dev portaudio19-dev i2c-tools libatlas-base-dev ffmpeg

# 2. Enable I2C and I2S Interfaces
echo "[*] Enabling I2C Interface..."
sudo raspi-config nonint do_i2c 0

# 3. Install Python Machine Learning & Audio Libraries
echo "[*] Installing Python Dependencies..."
pip3 install --upgrade pip
pip3 install numpy scipy pandas librosa matplotlib flask requests pillow
pip3 install adafruit-circuitpython-ssd1306 adafruit-blinka
pip3 install tflite-runtime

# 4. Create Systemd Service for Auto-Start on Boot
echo "[*] Creating Systemd Auto-Start Service..."
cat << 'EOF' | sudo tee /etc/systemd/system/cardiopulse.service
[Unit]
Description=CardioPulse AI Smart Stethoscope Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/THEJAS/rpi_stethoscope_oled.py
WorkingDirectory=/home/pi/THEJAS
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cardiopulse.service

echo ""
echo "=========================================================="
echo "  [+] Raspberry Pi Zero 2W Setup Completed Successfully!  "
echo "  To launch manually: python3 rpi_stethoscope_oled.py    "
echo "=========================================================="
