/*
 * Single Channel AD8232 - Phase 4 Subvocalization
 *
 * Uses ONE working AD8232 sensor for submental EMG capture.
 * Position electrodes under chin for subvocalization detection.
 *
 * Wiring:
 *   AD8232:  OUTPUT -> GPIO34, LO+ -> GPIO32, LO- -> GPIO33
 *            SDN -> 3.3V, GND -> GND, 3.3V -> 3.3V
 */

// ============ PIN DEFINITIONS ============
#define EMG_OUTPUT 34
#define LO_PLUS 32
#define LO_MINUS 33

// ============ CONFIGURATION ============
#define SAMPLE_RATE_HZ 500
#define BAUD_RATE 115200

const int sampleDelayMs = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  Serial.println("Single Channel Subvocal EMG - Phase 4");
  Serial.println("Format: EMG_Value,LeadsOff");
  Serial.println("---");
  delay(1000);
}

void loop() {
  // Check leads-off detection
  int lo_plus = digitalRead(LO_PLUS);
  int lo_minus = digitalRead(LO_MINUS);
  int leads_off = (lo_plus == 1 || lo_minus == 1) ? 1 : 0;

  // Read EMG signal (0-4095)
  int emg_value = analogRead(EMG_OUTPUT);

  // Output: EMG_Value,LeadsOff (0=connected, 1=leads off)
  Serial.print(emg_value);
  Serial.print(",");
  Serial.println(leads_off);

  delay(sampleDelayMs);
}
