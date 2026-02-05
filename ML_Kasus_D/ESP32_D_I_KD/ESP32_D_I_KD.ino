// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_d_4_int8.h"
#include "test_data_kasus_d.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define NUM_CLASSES 3
#define TF_NUM_OPS 5
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
// !!! PASTIKAN SEMUA NILAI INI SESUAI DENGAN MODEL ANDA !!!
const float input_scale = 0.03787701204419136;
const int input_zero_point = -53;
const float output_scale = 0.00390625; // GANTI NILAI INI
const int output_zero_point = -128;      // GANTI NILAI INI

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
    tf.resolver.AddSoftmax();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array INT8
    while (!tf.begin(g_kasus_d_model_4_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus D - Quantized with Dequant) ---");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-----------------------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus.
 */
void loop() {
    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // --- Kuantisasi Manual Input ---
        int8_t input_quantized[FEATURES_PER_SAMPLE_D];
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            float feature_float = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
            input_quantized[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }
        
        int expected_label = test_labels_d[i];

        // Jalankan inferensi dan ukur waktu
        unsigned long start_time = micros();
        if (!tf.predict(input_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // --- De-kuantisasi Output lalu Cari Probabilitas Tertinggi ---
        int predicted_label = -1;
        float max_prob = -1.0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            // Ambil output int8 dan ubah ke float (probabilitas)
            int8_t output_quantized = tf.output(j);
            float current_prob = (float)(output_quantized - output_zero_point) * output_scale;

            // Cari probabilitas tertinggi
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = j;
            }
        }
        
        // Cetak hasil per baris
        Serial.printf("%d\t%d\t%d\t\t%lu\n", i + 1, expected_label, predicted_label, latency);
    }
}