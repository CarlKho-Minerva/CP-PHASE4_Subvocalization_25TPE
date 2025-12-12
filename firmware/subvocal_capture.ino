// PHASE 4: Subvocalization - ESP32 Firmware
// Dual-channel AD8232 sEMG capture at 1000Hz

// ============================================
// PIN CONFIGURATION
// ============================================
#define AD8232_CH1    34    // Chin (Digastric) - Primary
#define AD8232_CH2    35    // Jaw (Masseter) - Secondary
#define SDN_PIN       25    // Shutdown pin - wire to 3.3V or set HIGH

#define SAMPLE_RATE   1000  // Hz
#define BAUD_RATE     115200

// ============================================
// TIMING
// ============================================
unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 1000 / SAMPLE_RATE;  // microseconds

// ============================================
// SETUP
// ============================================
void setup() {
  Serial.begin(BAUD_RATE);

  // Configure ADC
  analogReadResolution(12);  // 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db);  // Full 3.3V range

  // CRITICAL: SDN pin must be HIGH to enable AD8232
  pinMode(SDN_PIN, OUTPUT);
  digitalWrite(SDN_PIN, HIGH);

  // Status LED
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.println("# PHASE 4 Subvocalization - Dual Channel sEMG");
  Serial.println("# CH1=Chin, CH2=Jaw, Rate=1000Hz");
  Serial.println("# Format: timestamp,ch1,ch2");

  delay(500);  // Let AD8232 stabilize
}

// ============================================
// MAIN LOOP
// ============================================
void loop() {
  unsigned long currentTime = micros();

  if (currentTime - lastSampleTime >= sampleInterval * 1000) {
    lastSampleTime = currentTime;

    // Read both channels
    int ch1 = analogRead(AD8232_CH1);
    int ch2 = analogRead(AD8232_CH2);

    // Output CSV format: timestamp, channel1, channel2
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ch1);
    Serial.print(",");
    Serial.println(ch2);
  }
}

/*
WIRING REFERENCE:

ESP32           AD8232 (Channel 1 - Chin)
------          -------------------------
3.3V    ────────  3.3V
GND     ────────  GND
GPIO 34 ────────  OUTPUT
GPIO 25 ────────  SDN (or wire SDN directly to 3.3V)

ESP32           AD8232 (Channel 2 - Jaw)
------          ------------------------
3.3V    ────────  3.3V  (shared with CH1)
GND     ────────  GND   (shared with CH1)
GPIO 35 ────────  OUTPUT

ELECTRODE PLACEMENT:
- CH1: Under chin (Red+Yellow 2cm apart), Green on mastoid
- CH2: Masseter/jaw cheek, Green on mastoid or collarbone

⚠️ SAFETY: Run on battery power only!
*/
