#include "model_a_4_float.h" 
#include "test_data_kasus_a.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// Definisikan jumlah operator dan ukuran arena tensor
#define TF_NUM_OPS 6 
#define ARENA_SIZE 8000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH berarti mati untuk LED internal)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Konfigurasi input/output model [cite: 33]
    tf.setNumInputs(1);
    tf.setNumOutputs(1);
    
    // Daftarkan semua operator yang mungkin digunakan oleh model [cite: 34]
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu(); // [cite: 35]
    tf.resolver.AddReshape(); // [cite: 35]
    tf.resolver.AddQuantize(); // [cite: 35]
    tf.resolver.AddDequantize(); // [cite: 35]
    
    // Inisialisasi model dari C array [cite: 36]
    while (!tf.begin(g_kasus_a_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000); // [cite: 37]
    }
    
    // Fase Idle selama 5 detik
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);

    // Mencetak header tabel (opsional, untuk debugging)
    Serial.println("--- Continuous Inference Started (ESP32 - Case A) ---");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus.
 */
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_A; i++) {
        // Mengambil fitur input dan label yang diharapkan
        float x = test_features_a[i];
        float expected_y = test_labels_a[i]; // [cite: 23]
        
        // Menjalankan inferensi dan mengukur waktu
        unsigned long start_time = micros();
        if (!tf.predict(&x).isOk()) { // [cite: 24]
            Serial.println(tf.exception.toString());
            continue; // Lanjut ke data berikutnya jika ada error
        }
        unsigned long latency = micros() - start_time;
        
        // Membaca hasil prediksi
        float predicted_y = tf.output(); // [cite: 26]
        
        // Mencetak hasil per baris (opsional, untuk debugging)
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", x, expected_y, predicted_y, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}