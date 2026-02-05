#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>

// --- GANTI DENGAN KREDENSIAL WIFI ANDA ---
const char* ssid = "Xiaomi";
const char* password = "12345678";
// -----------------------------------------

// Nama file yang akan di-download dari SPIFFS
const char* namaFile = "/jog_0005.csv"; 

WebServer server(80); // Membuat objek server di port 80

void handleDownload() {
  // Cek apakah file yang diminta ada di SPIFFS
  if (SPIFFS.exists(namaFile)) {
    // Buka file untuk dibaca
    File file = SPIFFS.open(namaFile, "r");
    if (file) {
      // Mengirim header ke browser agar browser tahu ini adalah file untuk di-download
      server.sendHeader("Content-Disposition", "attachment; filename=" + String(namaFile).substring(1));
      
      // Streaming isi file ke browser. Ini efisien untuk file besar.
      server.streamFile(file, "text/plain"); 
      
      // Tutup file setelah selesai
      file.close();
      Serial.println("File telah dikirimkan.");
      return;
    }
  }
  // Jika file tidak ditemukan, kirim pesan error 404
  server.send(404, "text/plain", "404: File Not Found");
  Serial.println("Gagal: File tidak ditemukan di SPIFFS.");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Inisialisasi SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("Gagal me-mount SPIFFS!");
    return;
  }
  Serial.println("SPIFFS berhasil di-mount.");

  // Koneksi ke WiFi
  WiFi.begin(ssid, password);
  Serial.print("Menghubungkan ke WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  Serial.println("\nTerhubung!");
  Serial.print("Alamat IP ESP32: ");
  Serial.println(WiFi.localIP()); // Cetak alamat IP

  // Menentukan endpoint URL untuk download
  server.on("/download", HTTP_GET, handleDownload);
  
  // Memulai web server
  server.begin();
  Serial.println("Web server aktif. Buka http://<IP_Address>/download di browser.");
}

void loop() {
  // Menangani request yang masuk ke server
  server.handleClient(); 
}