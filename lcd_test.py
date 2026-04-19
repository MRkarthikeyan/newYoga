"""
LCD Diagnostic Test Script
Run this on the Raspberry Pi to isolate LCD issues from the main app.
Usage: python3 lcd_test.py
"""
import time

print("=" * 40)
print("  LCD DISPLAY DIAGNOSTIC TEST")
print("=" * 40)

# Step 1: Check library
print("\n[1] Checking rpi_lcd library...")
try:
    from rpi_lcd import LCD
    print("    ✅ rpi_lcd imported successfully")
except ImportError:
    print("    ❌ rpi_lcd NOT found!")
    print("    Fix: pip install rpi-lcd smbus2")
    exit(1)

# Step 2: Initialize
print("\n[2] Initializing LCD...")
try:
    lcd = LCD()
    print("    ✅ LCD initialized successfully!")
except Exception as e:
    print(f"    ❌ LCD init failed: {e}")
    print("\n    Possible causes:")
    print("    • I2C not enabled → run: sudo raspi-config → Interface Options → I2C → Yes")
    print("    • Wiring wrong   → VCC=5V, GND=GND, SDA=Pin3, SCL=Pin5")
    print("    • Wrong I2C bus  → try: sudo i2cdetect -y 0  (instead of -y 1)")
    exit(1)

# Step 3: Write to LCD
print("\n[3] Writing text to LCD...")
try:
    lcd.clear()
    time.sleep(0.1)
    lcd.text("LCD Test OK!", 1)
    lcd.text("Yoga Judge", 2)
    print("    ✅ Text written to LCD!")
    print("\n    >>> CHECK YOUR LCD SCREEN NOW <<<")
    print("    If the screen is blank (backlight on but no text):")
    print("    → Turn the small blue CONTRAST SCREW on the back of the LCD")
    print("      slowly with a screwdriver until text appears.\n")
except Exception as e:
    print(f"    ❌ Write failed: {e}")
    exit(1)

# Step 4: Hold for viewing
print("[4] Holding display for 5 seconds...")
time.sleep(5)

# Step 5: Test all messages the yoga app will show
print("\n[5] Testing all yoga app messages...")

messages = [
    ("Enter into frame", "Yoga Judge Ready"),
    ("Analysing...",     "Hold for 5s"),
    ("Warrior II",       "Score: 8/10"),
    ("No Pose Detected", "Score: 0/10"),
    ("Enter into frame", "Competitor #2"),
]

for line1, line2 in messages:
    print(f"    Showing: '{line1}' / '{line2}'")
    lcd.clear()
    time.sleep(0.05)
    lcd.text(line1[:16], 1)
    lcd.text(line2[:16], 2)
    time.sleep(2)

lcd.clear()
print("\n✅ ALL TESTS PASSED — LCD is working correctly!")
print("   The yoga app LCD should now work as expected.\n")
