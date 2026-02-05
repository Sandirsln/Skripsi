// 1. SERTAKAN HEADER MODEL & DATA UJI (DIUBAH KE KASUS E)
#include "model_e_4_float.h" // <-- GANTI: Model Floating Point Kasus E Anda
#include "test_data_kasus_e.h" // <-- GANTI: Data Uji Kasus E Anda

// 2. SERTAKAN PUSTAKA TFLM (TETAP)
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL (TETAP)
#define NUM_CLASSES 3
#define TF_NUM_OPS 6
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (LOW/ACTIVE HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Konfigurasi input/output model (DIUBAH KE KONSTANTA KASUS E)
    tf.setNumInputs(FEATURES_PER_SAMPLE_E); // <-- DIGANTI
    tf.setNumOutputs(NUM_CLASSES);

    // Daftarkan operator yang dibutuhkan (TETAP)
    // Walaupun model float, AddQuantize/AddDequantize mungkin diperlukan
    // jika model float Anda memiliki operator yang awalnya didukung oleh kuantisasi.
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddSoftmax();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array (DIUBAH KE VARIABEL KASUS E)
    // Pastikan nama variabel C array model sesuai dengan file header Anda
    while (!tf.begin(g_kasus_e_model_4_float_model_data).isOk()) { // <-- DIGANTI
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik untuk pengukuran daya baseline
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus E - Float) ---");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus untuk inferensi.
 */
void loop() {
    // Mengulang untuk semua data tes di dalam loop (DIUBAH KE KONSTANTA KASUS E)
    for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) { // <-- DIGANTI
        
        // Buat buffer sementara untuk menampung fitur input (DIUBAH KE KONSTANTA KASUS E)
        float input_sample_buffer[FEATURES_PER_SAMPLE_E]; // <-- DIGANTI
        
        // Mengisi buffer dengan data fitur Kasus E
        for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++) { // <-- DIGANTI
            input_sample_buffer[j] = test_features_e[i * FEATURES_PER_SAMPLE_E + j]; // <-- DIGANTI
        }
        
        // Mengambil label yang diharapkan dari data uji Kasus E
        int expected_label = test_labels_e[i]; // <-- DIGANTI

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_time = micros();
        if (!tf.predict(input_sample_buffer).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // --- Logika Prediksi untuk Multikelas (TETAP SAMA) ---
        // Cari indeks (kelas) dari output dengan probabilitas tertinggi
        int predicted_label = -1;
        float max_prob = -1.0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (tf.output(j) > max_prob) {
                max_prob = tf.output(j);
                predicted_label = j;
            }
        }
        
        // Cetak hasil per baris
        Serial.printf("%d\t%d\t%d\t\t%lu\n", i + 1, expected_label, predicted_label, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}