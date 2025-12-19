/*
 * AD8232 Diagnostic Test - Is My Board Dead?
 *
 * Minimal test to verify AD8232 functionality.
 * Uses NodeMCU-32S pinout from your guide.
 *
 * Wiring:
 *   AD8232 OUTPUT → GPIO36 (pin 5)
 *   AD8232 LO+    → GPIO26 (pin 15)
 *   AD8232 LO-    → GPIO25 (pin 14)
 *   AD8232 SDN    → 3.3V (CRITICAL!)
 *   AD8232 3.3V   → 3.3V
 *   AD8232 GND    → GND
 */

#define AD8232_OUTPUT 36
#define LO_PLUS 26
#define LO_MINUS 25

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  Serial.println();
  Serial.println("=== AD8232 DIAGNOSTIC TEST ===");
  Serial.println("Testing if your AD8232 is alive...");
  Serial.println();
  delay(1000);
}

void loop() {
  int lo_plus = digitalRead(LO_PLUS);
  int lo_minus = digitalRead(LO_MINUS);
  int adc_value = analogRead(AD8232_OUTPUT);

  // Print diagnostic info
  Serial.print("ADC: ");
  Serial.print(adc_value);
  Serial.print(" | LO+: ");
  Serial.print(lo_plus);
  Serial.print(" | LO-: ");
  Serial.print(lo_minus);
  Serial.print(" | Status: ");

  // Diagnostic logic
  if (adc_value < 100) {
    Serial.println("⚠️  ADC LOW - Check OUTPUT wiring or SDN pin!");
  } else if (adc_value > 3900) {
    Serial.println("⚠️  ADC RAILING HIGH - Possible SDN issue or dead IC");
  } else if (lo_plus == 1 && lo_minus == 1) {
    Serial.println("✓ Board responding (leads off - expected without electrodes)");
  } else {
    Serial.println("✓ Board ALIVE - getting signal!");
  }

  delay(500);
}
