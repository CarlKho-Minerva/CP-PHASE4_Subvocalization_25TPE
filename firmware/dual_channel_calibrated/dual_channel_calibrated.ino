/*
 * Dual Channel AD8232 - Phase 4 Subvocalization
 * With INDEPENDENT BASELINE CALIBRATION per sensor
 *
 * Wiring (ESP32 DevKit):
 *   AD8232 #1 (Chin): OUTPUT→GPIO34, LO+→GPIO32, LO-→GPIO33
 *   AD8232 #2 (Jaw):  OUTPUT→GPIO36, LO+→GPIO25, LO-→GPIO26
 *   Both: SDN→3.3V, GND→GND, 3.3V→3.3V
 */

// ============ PIN DEFINITIONS ============
// Sensor 1 (Chin)
#define S1_OUTPUT 34
#define S1_LO_PLUS 32
#define S1_LO_MINUS 33

// Sensor 2 (Jaw)
#define S2_OUTPUT 36
#define S2_LO_PLUS 25
#define S2_LO_MINUS 26

// ============ CONFIGURATION ============
#define SAMPLE_RATE_HZ 500
#define BAUD_RATE 115200
#define CALIBRATION_SAMPLES 200  // ~1 second of samples for baseline

// Baselines (calibrated on startup)
int baseline1 = 2048;  // Will be calibrated
int baseline2 = 2048;  // Will be calibrated

const int sampleDelayMs = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(S1_LO_PLUS, INPUT);
  pinMode(S1_LO_MINUS, INPUT);
  pinMode(S2_LO_PLUS, INPUT);
  pinMode(S2_LO_MINUS, INPUT);

  Serial.println("Dual Channel Subvocal EMG - Phase 4");
  Serial.println("Calibrating baselines... RELAX YOUR FACE!");
  delay(2000);  // Give user time to relax

  calibrateBaselines();

  Serial.println("---");
  Serial.print("Baseline1 (Chin): "); Serial.println(baseline1);
  Serial.print("Baseline2 (Jaw):  "); Serial.println(baseline2);
  Serial.println("---");
  Serial.println("Format: S1_Deviation,S2_Deviation,S1_LO,S2_LO");
  delay(1000);
}

void calibrateBaselines() {
  long sum1 = 0, sum2 = 0;

  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    sum1 += analogRead(S1_OUTPUT);
    sum2 += analogRead(S2_OUTPUT);
    delay(5);
  }

  baseline1 = sum1 / CALIBRATION_SAMPLES;
  baseline2 = sum2 / CALIBRATION_SAMPLES;
}

void loop() {
  // Lead-off detection (combined per sensor)
  int s1_leads_off = (digitalRead(S1_LO_PLUS) == 1 || digitalRead(S1_LO_MINUS) == 1) ? 1 : 0;
  int s2_leads_off = (digitalRead(S2_LO_PLUS) == 1 || digitalRead(S2_LO_MINUS) == 1) ? 1 : 0;

  // Read raw values
  int raw1 = analogRead(S1_OUTPUT);
  int raw2 = analogRead(S2_OUTPUT);

  // Calculate deviation from baseline (centered around 0)
  int dev1 = raw1 - baseline1;
  int dev2 = raw2 - baseline2;

  // Output: Deviation1,Deviation2,LeadsOff1,LeadsOff2
  // Deviations will be +/- values around 0
  // Muscle activity = large positive or negative spikes
  Serial.print(dev1);
  Serial.print(",");
  Serial.print(dev2);
  Serial.print(",");
  Serial.print(s1_leads_off);
  Serial.print(",");
  Serial.println(s2_leads_off);

  delay(sampleDelayMs);
}
