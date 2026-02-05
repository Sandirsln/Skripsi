#include "FastIMU.h"
#include <Wire.h>
#include "FS.h"
#include "SPIFFS.h"
#include <Preferences.h>

// ---- Konfigurasi ----
#define IMU_ADDRESS       0x68
static const int SR_HZ        = 50;   // 50 Hz
static const int DURATION_SEC = 300;  // 5 menit
// ---------------------

MPU6050 IMU;
calData  calib = {0};
AccelData accelData;
GyroData  gyroData;
Preferences prefs;

// Buat nama file unik per sesi: /idle_0001.csv, /idle_0002.csv, ...
static String makeSessionFilename(const char* activity = "idle") {
  prefs.begin("log", false);
  uint32_t n = prefs.getUInt("sess", 0) + 1;
  prefs.putUInt("sess", n);
  prefs.end();
  char buf[32];
  snprintf(buf, sizeof(buf), "/%s_%04lu.csv", activity, (unsigned long)n);
  return String(buf);
}

static void loadCalibration() {
  // Ganti jika punya hasil kalibrasi baru
  calib.accelBias[0] =  1.03f;
  calib.accelBias[1] = -0.02f;
  calib.accelBias[2] = -0.92f;
  calib.gyroBias[0]  = -3.08f;
  calib.gyroBias[1]  =  0.37f;
  calib.gyroBias[2]  = -0.15f;
  calib.magBias[0] = calib.magBias[1] = calib.magBias[2] = 0.0f;
  calib.magScale[0] = calib.magScale[1] = calib.magScale[2] = 1.0f;
}

static bool initIMU() {
  Wire.begin();            // ESP32 default: SDA=21, SCL=22
  Wire.setClock(400000);   // 400 kHz
  if (IMU.init(calib, IMU_ADDRESS) != 0) return false;
  IMU.setGyroRange(500);   // ±500 dps
  IMU.setAccelRange(4);    // ±4 g
  return true;
}

void setup() {
  loadCalibration();

  // Mount SPIFFS TANPA format otomatis; jika gagal, hentikan (fail-safe)
  if (!SPIFFS.begin(false)) { return; }

  // Inisialisasi IMU; jika gagal, hentikan (fail-safe)
  if (!initIMU()) { return; }

  // Buat file CSV baru (unik per sesi)
  String filename = makeSessionFilename("jog");
  File f = SPIFFS.open(filename, FILE_WRITE);
  if (!f) { return; }

  // Header CSV (komentar diawali #)
  f.println("# activity=jogging");
  f.printf("# sr_hz=%d\n", SR_HZ);
  f.println("# mount=hip_bag");
  f.println("# units: accel=g, gyro=dps");
  f.printf("# accel_bias=%.3f,%.3f,%.3f\n", calib.accelBias[0], calib.accelBias[1], calib.accelBias[2]);
  f.printf("# gyro_bias=%.3f,%.3f,%.3f\n",   calib.gyroBias[0],  calib.gyroBias[1],  calib.gyroBias[2]);
  f.printf("# filename=%s\n", filename.c_str());
  f.println("timestamp_ms,accX,accY,accZ,gyroX,gyroY,gyroZ");
  f.flush();

  // Logging 50 Hz selama 5 menit
  const uint32_t totalSamples = (uint32_t)SR_HZ * (uint32_t)DURATION_SEC; // 15000
  const uint32_t period_us    = 1000000UL / (uint32_t)SR_HZ;              // 20000 us
  uint64_t nextTick = micros();
  uint32_t written  = 0;
  uint32_t lastFlush= 0;

  while (written < totalSamples) {
    uint64_t now_us = micros();
    if (now_us < nextTick) {
      uint32_t rem = (uint32_t)(nextTick - now_us);
      if (rem > 200) delayMicroseconds(rem - 100);
      continue;
    }
    nextTick += period_us;

    IMU.update();
    IMU.getAccel(&accelData);
    IMU.getGyro(&gyroData);

    unsigned long tms = millis();  // timestamp ms sejak boot
    f.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
             tms,
             accelData.accelX, accelData.accelY, accelData.accelZ,
             gyroData.gyroX,  gyroData.gyroY,  gyroData.gyroZ);

    written++;
    if ((written - lastFlush) >= 150) { f.flush(); lastFlush = written; }
  }

  f.println("# END");
  f.flush();
  f.close();

  // Headless selesai — aman dicabut dayanya.
}

void loop() {
  // kosong
}
