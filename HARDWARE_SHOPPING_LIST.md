# Hardware Shopping List - Guang Hua Digital Plaza

## Essential Components

### 1. Sensor: AD8232 Module (x2) - ~$150-300 TWD each
- **Look for:** Red or purple breakout board labeled "AD8232 Single Lead Heart Rate Monitor"
- **Check:** Must include **3-lead cable** (black, ends in 3.5mm jack)
- **Why x2:** One for chin (articulation), one for jaw (intensity) = doubled accuracy

### 2. Electrodes: Disposable ECG/EMG Pads (x50 pack)
- **Specifics:** "Ag/AgCl" foam pads with metal snap
- **Why:** Need conductive gel - dry copper/wires won't work
- **⚠️ Critical:** If they don't have sticky pads, DON'T buy the sensor

### 3. Microcontroller: ESP32 Development Board
- **Recommended:** NodeMCU-32S or similar
- **Why ESP32:**
  1. Much faster than Arduino (essential for signal processing)
  2. 3.3V logic matches AD8232 perfectly
  3. Built-in Bluetooth/WiFi for wireless later

### 4. Signal Quality "Secret Sauce"
- **Shielded Cable / Audio Cable**
  - Problem: Stock 3-lead wires act as antennas for noise
  - Fix: Buy **3.5mm audio extension cable** to cut open
  - Alternative: Braid wires together + wrap in foil/tape

### 5. Power: USB Power Bank (battery)
- **⚠️ CRITICAL SAFETY:**
  - NEVER connect electrodes while laptop plugged into wall
  - 60Hz mains hum destroys signal AND is safety risk
  - Must use battery power for laptop OR USB power bank for ESP32

## Optional But Recommended

| Item | Purpose |
|------|---------|
| Alcohol swabs | Skin prep for better contact |
| Medical tape | Secure electrodes during jaw movement |
| Gold-plated header pins | Board connections |
| Snap connectors | If cutting stock cable |

## Where to Go

**Jin Hua (今華)** or **Guang Hua B1/top floors**
→ Go to component shops, NOT consumer laptop stores
→ Show them the pictures if needed

## Wiring Reference

```
ESP32           AD8232
------          ------
3.3V    ───────  3.3V
GND     ───────  GND
GPIO 34 ───────  OUTPUT
        ───────  SDN → 3.3V (CRITICAL: Don't leave floating!)
```

For 2nd sensor: Use GPIO 36 (VP) or GPIO 39 (VN)

## Cost Breakdown

| Item | Est. Cost (TWD) |
|------|-----------------|
| AD8232 x2 | 300-600 |
| Electrodes x50 | 200 |
| ESP32 | 200 |
| Cables/misc | 100 |
| Power bank | 300 |
| **Total** | **~1,100-1,400 TWD (~$35-45 USD)** |

---

*Remember: Test heartbeat first, then jaw clench, then subvocalization.*
