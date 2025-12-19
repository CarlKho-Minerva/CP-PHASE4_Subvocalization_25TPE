#!/usr/bin/env python3
"""
Tap-to-Capture EMG - Phase 4 Subvocalization
Tap SPACE = Record exactly 1 second of data
"""

import serial
import serial.tools.list_ports
import time
import csv
import os
import sys

try:
    from pynput import keyboard
except ImportError:
    os.system(f"{sys.executable} -m pip install pynput")
    from pynput import keyboard

# Configuration
BAUD_RATE = 230400
OUTPUT_DIR = "data"
RECORD_DURATION = 1.0
LABELS = ["GHOST", "LEFT", "STOP", "REST"]
LEVELS = ["L1_Overt", "L2_Whisper", "L3_Mouthing", "L4_Silent"]

current_label = "REST"
current_level = "L3_Mouthing"
sample_count = {label: {level: 0 for level in LEVELS} for label in LABELS}
ser = None
is_recording = False


def list_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def select_port():
    ports = list_ports()
    if not ports:
        print("No serial ports found!")
        return None

    # Filter for USB ports
    usb_ports = [p for p in ports if 'usb' in p.lower()]
    if len(usb_ports) == 1:
        print(f"Auto-selecting {usb_ports[0]}")
        return usb_ports[0]

    print("\nAvailable ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port}")

    while True:
        try:
            selection = int(input("Select port: "))
            if 0 <= selection < len(ports):
                return ports[selection]
        except ValueError:
            pass


def print_status(message=""):
    os.system('clear' if os.name != 'nt' else 'cls')
    print("=" * 50)
    print("  SUBVOCAL CAPTURE - Tap SPACE for 1s window")
    print("=" * 50)
    print(f"\n  Label: [{current_label}]  Level: [{current_level}]")
    print("\n  G/L/S/R=Label | 1-4=Level | ESC=Quit")
    print("-" * 50)
    for label in LABELS:
        counts = [f"{sample_count[label][l]}" for l in LEVELS]
        marker = "→" if label == current_label else " "
        print(f"  {marker} {label}: L1={counts[0]} L2={counts[1]} L3={counts[2]} L4={counts[3]}")
    print("-" * 50)
    if message:
        print(f"\n  {message}")
    else:
        print(f"\n  Tap SPACE to record")


def record_one_second():
    global is_recording

    if is_recording:
        return
    is_recording = True

    print_status(f"🔴 RECORDING {current_label}...")

    # Flush and wait a tiny bit
    ser.reset_input_buffer()
    time.sleep(0.05)

    samples = []
    start_time = time.time()
    raw_bytes = b""

    # Record for 1 second, reading raw bytes
    while (time.time() - start_time) < RECORD_DURATION:
        if ser.in_waiting > 0:
            raw_bytes += ser.read(ser.in_waiting)

    # Parse lines from raw bytes
    try:
        text = raw_bytes.decode('utf-8', errors='ignore')
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    ts, val = parts[0].strip(), parts[1].strip()
                    if ts.isdigit() and (val.isdigit() or val.lstrip('-').isdigit()):
                        samples.append([ts, val])
    except Exception as e:
        print(f"Parse error: {e}")

    # Save
    if samples:
        sample_count[current_label][current_level] += 1
        count = sample_count[current_label][current_level]

        level_dir = os.path.join(OUTPUT_DIR, current_level)
        os.makedirs(level_dir, exist_ok=True)

        filename = os.path.join(level_dir, f"{current_label}_{count:02d}.csv")

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "RawValue"])
            writer.writerows(samples)

        print_status(f"✅ {len(samples)} samples → {current_label}_{count:02d}.csv")
    else:
        print_status(f"⚠️ No samples! Raw bytes: {len(raw_bytes)}")

    is_recording = False


def on_press(key):
    global current_label, current_level

    try:
        if hasattr(key, 'char') and key.char:
            c = key.char.lower()
            if c == 'g': current_label = "GHOST"; print_status()
            elif c == 'l': current_label = "LEFT"; print_status()
            elif c == 's': current_label = "STOP"; print_status()
            elif c == 'r': current_label = "REST"; print_status()
            elif c == '1': current_level = "L1_Overt"; print_status()
            elif c == '2': current_level = "L2_Whisper"; print_status()
            elif c == '3': current_level = "L3_Mouthing"; print_status()
            elif c == '4': current_level = "L4_Silent"; print_status()

        if key == keyboard.Key.space:
            record_one_second()

        if key == keyboard.Key.esc:
            return False
    except Exception as e:
        print(f"Key error: {e}")


def main():
    global ser

    print("\n  Electrode: Under chin, 2-3cm apart")
    print("  Reference: Behind ear\n")

    port = select_port()
    if not port:
        return

    print(f"\nConnecting to {port} @ {BAUD_RATE}...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
        time.sleep(2)  # Wait for ESP32 reset
        ser.reset_input_buffer()

        # Quick test read
        print("Testing connection...", end=" ", flush=True)
        time.sleep(0.3)
        test = ser.read(200)
        print(f"Got {len(test)} bytes")
        if len(test) > 0:
            print(f"Sample: {test[:50]}")

        time.sleep(0.5)
        print_status()

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
        print("\n\nDone!")


if __name__ == "__main__":
    main()
