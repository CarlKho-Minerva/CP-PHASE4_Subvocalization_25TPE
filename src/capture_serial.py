"""
PHASE 4: Subvocalization - Serial Data Capture
Reads dual-channel sEMG from ESP32 and saves to CSV
"""

import serial
import csv
import time
from datetime import datetime
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
SERIAL_PORT = "/dev/tty.usbserial-0001"  # Update for your ESP32
BAUD_RATE = 115200
OUTPUT_DIR = Path("../data/raw")

# ============================================
# MAIN
# ============================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"session_{timestamp}.csv"

    print(f"[*] Connecting to {SERIAL_PORT}...")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for ESP32 to stabilize

        print(f"[+] Connected! Recording to {output_file}")
        print("[i] Press Ctrl+C to stop recording\n")

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ch1_chin', 'ch2_jaw'])

            sample_count = 0
            start_time = time.time()

            while True:
                line = ser.readline().decode('utf-8').strip()

                # Skip comment lines
                if line.startswith('#') or not line:
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) == 3:
                        writer.writerow(parts)
                        sample_count += 1

                        # Progress update every second
                        if sample_count % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = sample_count / elapsed
                            print(f"\r[{sample_count:,} samples | {rate:.0f} Hz] {line}", end='')
                except:
                    pass

    except KeyboardInterrupt:
        print(f"\n\n[+] Recording stopped. Saved {sample_count:,} samples to {output_file}")
    except serial.SerialException as e:
        print(f"[!] Serial error: {e}")
        print("[i] Check your SERIAL_PORT setting and that ESP32 is connected")

if __name__ == "__main__":
    main()
