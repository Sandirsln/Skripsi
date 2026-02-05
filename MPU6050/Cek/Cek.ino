#include "FS.h"
#include "SPIFFS.h"

void setup() {
  Serial.begin(115200);
  delay(500);

  if (!SPIFFS.begin(false)) {  // pakai 'false' agar tidak format otomatis
    Serial.println("[SPIFFS] Gagal mount. Pastikan partisi benar dan tidak corrupt.");
    return;
  }

  Serial.println("\n=== Daftar File di SPIFFS ===");
  listSPIFFS();
  Serial.println("==============================");

  // Opsional: tampilkan info memori
  size_t total = SPIFFS.totalBytes();
  size_t used  = SPIFFS.usedBytes();
  Serial.printf("Total: %u bytes | Digunakan: %u bytes | Sisa: %u bytes\n",
                (unsigned)total, (unsigned)used, (unsigned)(total - used));
}

void loop() {
  // tidak ada yang dilakukan
}

void listSPIFFS() {
  File root = SPIFFS.open("/");
  if (!root) {
    Serial.println("[SPIFFS] Gagal membuka root directory!");
    return;
  }

  File file = root.openNextFile();
  if (!file) {
    Serial.println("[SPIFFS] Tidak ada file ditemukan.");
    return;
  }

  while (file) {
    Serial.printf("📄 %s\t(%u bytes)\n", file.name(), (unsigned)file.size());
    file = root.openNextFile();
  }
}
