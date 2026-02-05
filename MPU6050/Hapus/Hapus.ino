#include "FS.h"
#include "SPIFFS.h"
#include <vector>

void setup() {
  Serial.begin(115200);
  delay(400);

  if (!SPIFFS.begin(false)) {
    Serial.println("[SPIFFS] Gagal mount.");
    return;
  }

  Serial.println("=== Menghapus semua file di SPIFFS ===");
  std::vector<String> paths;

  // 1) Kumpulkan semua path file (dengan leading '/')
  {
    File root = SPIFFS.open("/");
    if (!root) {
      Serial.println("[SPIFFS] Tidak bisa buka root.");
      return;
    }
    File f = root.openNextFile();
    if (!f) {
      Serial.println("[SPIFFS] Tidak ada file.");
    }
    while (f) {
      String p = f.name();               // biasanya sudah "/nama.csv"
      if (!p.startsWith("/")) p = "/" + p; // pastikan ada '/'
      paths.push_back(p);
      // penting: tutup sebelum lanjut
      f.close();
      f = root.openNextFile();
    }
  }

  // 2) Hapus satu per satu dengan verifikasi exists()
  if (paths.empty()) {
    Serial.println("[SPIFFS] Tidak ada file untuk dihapus.");
  } else {
    for (auto &p : paths) {
      bool ex = SPIFFS.exists(p);
      Serial.printf("• %s  (%s)\n", p.c_str(), ex ? "ada" : "tidak ada");
      if (ex) {
        bool ok = SPIFFS.remove(p);
        Serial.printf("%s %s\n", ok ? "🗑️  Hapus:" : "❌ Gagal hapus:", p.c_str());
      }
    }
  }

  // 3) Tampilkan sisa file (seharusnya kosong)
  Serial.println("\n=== Sisa file setelah penghapusan ===");
  {
    bool any = false;
    File root = SPIFFS.open("/");
    File f = root.openNextFile();
    while (f) {
      any = true;
      Serial.printf("📄 %s (%u bytes)\n", f.name(), (unsigned)f.size());
      f.close();
      f = root.openNextFile();
    }
    if (!any) Serial.println("[SPIFFS] Tidak ada file tersisa.");
  }

  size_t total = SPIFFS.totalBytes();
  size_t used  = SPIFFS.usedBytes();
  Serial.printf("\nTotal: %u | Digunakan: %u | Sisa: %u bytes\n",
                (unsigned)total, (unsigned)used, (unsigned)(total - used));
}

void loop() {}
