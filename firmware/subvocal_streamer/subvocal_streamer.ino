/*
 * Single Channel Subvocal EMG Streamer - Phase 4
 *
 * Based on Phase 3 firmware, adapted for single-channel subvocalization.
 * 1000Hz sampling @ 230400 baud for high-speed Python capture.
 *
 * ELECTRODE PLACEMENT (From AlterEgo/Kapur 2018):
 * ================================================
 * For single channel, place electrodes in the CHIN/HYOID region:
 *
 *   Signal+ (Red/Yellow) → Under chin, LEFT of centerline
 *   Signal- (Yellow/Red) → Under chin, RIGHT of centerline (2-3cm apart)
 *   Reference (Green)    → Mastoid process (bony area behind ear)
 *
 * WHY THIS LOCATION?
 * - AlterEgo ranked "Mental" (chin tip) and "Hyoid" (under-chin) as TOP sites
 * - This captures Digastric/Mylohyoid muscles = TONGUE movement
 * - Tongue position changes for GHOST (back), LEFT (tip), STOP (behind teeth)
 * - Single channel here gives best signal-to-noise for word discrimination
 *
 * Wiring:
 *   AD8232 OUTPUT → GPIO36 (ADC1_CH0, input-only, cleanest)
 *   AD8232 SDN    → GPIO27 (set HIGH to enable)
 *   AD8232 3.3V   → 3.3V
 *   AD8232 GND    → GND
 */

#define AD8232_OUTPUT 36
#define SDN_PIN 27
#define LED_PIN 2

void setup() {
  // High-speed serial: 230400 baud for ~1000Hz data stream
  Serial.begin(230400);

  // Enable the AD8232
  pinMode(SDN_PIN, OUTPUT);
  digitalWrite(SDN_PIN, HIGH);

  // Activity LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MICROS = 1000; // 1000us = 1ms = 1000Hz

void loop() {
  unsigned long currentMicros = micros();

  // Precise 1000Hz sampling
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL_MICROS) {
    lastSampleTime = currentMicros;

    // Read raw EMG value (0-4095)
    int rawValue = analogRead(AD8232_OUTPUT);

    // Send: Timestamp(ms),RawValue
    Serial.print(millis());
    Serial.print(",");
    Serial.println(rawValue);
  }
}
