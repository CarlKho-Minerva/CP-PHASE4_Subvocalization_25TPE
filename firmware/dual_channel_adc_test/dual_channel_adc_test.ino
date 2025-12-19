/*
 * Dual Channel AD8232 Test - Phase 4 Subvocalization Project
 *
 * Reads two AD8232 sensors and outputs to Serial Plotter
 * Use Arduino IDE: Tools > Serial Plotter (Ctrl+Shift+L)
 *
 * Wiring:
 *   AD8232 #1 (Chin):  OUTPUT -> GPIO34, LO+ -> GPIO32, LO- -> GPIO33
 *   AD8232 #2 (Jaw):   OUTPUT -> GPIO36, LO+ -> GPIO25, LO- -> GPIO26
 *   Both: SDN -> 3.3V, GND -> GND, 3.3V -> 3.3V
 */

// ============ PIN DEFINITIONS ============
// Sensor 1 (Chin)
#define SENSOR1_OUTPUT 34
#define SENSOR1_LO_PLUS 32
#define SENSOR1_LO_MINUS 33

// Sensor 2 (Jaw)
#define SENSOR2_OUTPUT 36
#define SENSOR2_LO_PLUS 25
#define SENSOR2_LO_MINUS 26

// ============ CONFIGURATION ============
#define SAMPLE_RATE_HZ 500   // Samples per second
#define BAUD_RATE 115200     // Serial baud rate

// Calculate delay between samples
const int sampleDelayMs = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(BAUD_RATE);

  // Configure leads-off detection pins as inputs
  pinMode(SENSOR1_LO_PLUS, INPUT);
  pinMode(SENSOR1_LO_MINUS, INPUT);
  pinMode(SENSOR2_LO_PLUS, INPUT);
  pinMode(SENSOR2_LO_MINUS, INPUT);

  // Brief startup message
  Serial.println("Dual AD8232 Test - Phase 4 Subvocalization");
  Serial.println("Open Serial Plotter (Ctrl+Shift+L) to view graphs");
  Serial.println("---");
  delay(1000);
}

void loop() {
  // Read leads-off detection pins individually for debugging
  int s1_lo_plus = digitalRead(SENSOR1_LO_PLUS);
  int s1_lo_minus = digitalRead(SENSOR1_LO_MINUS);
  int s2_lo_plus = digitalRead(SENSOR2_LO_PLUS);
  int s2_lo_minus = digitalRead(SENSOR2_LO_MINUS);

  // Read raw ADC values (0-4095 for ESP32 12-bit ADC)
  int sensor1_value = analogRead(SENSOR1_OUTPUT);
  int sensor2_value = analogRead(SENSOR2_OUTPUT);

  // DEBUG OUTPUT: S1_Raw,S2_Raw,S1_LO+,S1_LO-,S2_LO+,S2_LO-
  // LO pins: 0 = electrode connected, 1 = leads off
  Serial.print(sensor1_value);
  Serial.print(",");
  Serial.print(sensor2_value);
  Serial.print(",");
  Serial.print(s1_lo_plus);    // Should be 0 when connected
  Serial.print(",");
  Serial.print(s1_lo_minus);   // Should be 0 when connected
  Serial.print(",");
  Serial.print(s2_lo_plus);    // Should be 0 when connected
  Serial.print(",");
  Serial.println(s2_lo_minus); // Should be 0 when connected

  delay(sampleDelayMs);
}
