/*
 * Single Channel AD8232 - Phase 4 Subvocalization
 * WITH BASELINE CALIBRATION
 *
 * Uses ONE working AD8232 sensor (the 1.8K baseline one).
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
#define CALIBRATION_SAMPLES 200

int baseline = 2048;
const int sampleDelayMs = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  Serial.println("Single Channel Subvocal EMG - Phase 4");
  Serial.println("Calibrating... RELAX YOUR FACE!");
  delay(2000);

  calibrateBaseline();

  Serial.print("Baseline: "); Serial.println(baseline);
  Serial.println("Format: Deviation,LeadsOff");
  Serial.println("---");
  delay(500);
}

void calibrateBaseline() {
  long sum = 0;
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    sum += analogRead(EMG_OUTPUT);
    delay(5);
  }
  baseline = sum / CALIBRATION_SAMPLES;
}

void loop() {
  int leads_off = (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) ? 1 : 0;

  int raw = analogRead(EMG_OUTPUT);
  int deviation = raw - baseline;  // Centered around 0

  // Output: Deviation,LeadsOff
  Serial.print(deviation);
  Serial.print(",");
  Serial.println(leads_off);

  delay(sampleDelayMs);
}
