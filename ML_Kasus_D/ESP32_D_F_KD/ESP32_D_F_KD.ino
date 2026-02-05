// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_d_4_float.h"
#include "test_data_kasus_d.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define NUM_CLASSES 3
#define TF_NUM_OPS 6
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Konfigurasi input/output model
    tf.setNumInputs(FEATURES_PER_SAMPLE_D);
    tf.setNumOutputs(NUM_CLASSES);

    // Daftarkan operator yang dibutuhkan
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddSoftmax(); // Menggunakan AddSoftmax untuk output multikelas
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array
    while (!tf.begin(g_kasus_d_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik untuk pengukuran daya baseline
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus D) ---");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus untuk inferensi.
 */
void loop() {
    // Mengulang untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // Buat buffer sementara untuk menampung beberapa fitur input
        float input_sample_buffer[FEATURES_PER_SAMPLE_D];
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            input_sample_buffer[j] = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
        }
        
        int expected_label = test_labels_d[i];

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_time = micros();
        if (!tf.predict(input_sample_buffer).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // --- Logika Prediksi untuk Multikelas ---
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