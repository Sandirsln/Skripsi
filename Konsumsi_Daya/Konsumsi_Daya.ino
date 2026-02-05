#include <Wire.h>
#include <Adafruit_INA219.h>

// Pin Definition untuk I2C1 di RP2350
#define I2C1_SDA_PIN 6
#define I2C1_SCL_PIN 7

Adafruit_INA219 ina219(0x40);

void setup() {
  Serial.begin(115200);
  delay(3000); // Delay penting untuk Serial USB RP2350

  // KONFIGURASI I2C1 (Wire1):
  // 1. Set pin ke instance Wire1 (karena GPIO 6/7 adalah milik I2C1)
  Wire1.setSDA(I2C1_SDA_PIN);
  Wire1.setSCL(I2C1_SCL_PIN);
  
  // 2. Mulai Wire1
  Wire1.begin(); 

  // 3. Pass pointer &Wire1 ke fungsi begin library
  // Ini memberitahu library untuk menggunakan port I2C1, bukan default I2C0
  if (!ina219.begin(&Wire1)) {
    Serial.println("Gagal mendeteksi INA219 pada Wire1. Periksa kabel!");
    while (1) { delay(10); }
  }

  Serial.println("INA219 Siap di I2C1.");
  Serial.println("Tegangan Bus (V), Arus (mA), Daya (mW)");
}

void loop() {
  float busVoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW   = ina219.getPower_mW();

  Serial.print(busVoltage, 3);
  Serial.print(", ");
  Serial.print(current_mA, 2);
  Serial.print(", ");
  Serial.println(power_mW, 2);

  delay(100);
}