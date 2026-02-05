/**
 * @file main_int8_d.cpp
 * @brief Benchmark model klasifikasi multikelas INT8 (Kasus D) pada ESP32.
 * * !!! LANGKAH WAJIB SEBELUM UPLOAD !!!
 * 1. Ganti file header model (misal: ke model_d_2_int8.h).
 * 2. SALIN & TEMPEL nilai 'scale' dan 'zero_point' dari output skrip Python
 * ke dalam konstanta di bawah ini.
 * 3. Sesuaikan nama variabel model di dalam tf.begin().
 */

// 1. SERTAKAN HEADER MODEL & DATA UJI
// =======================================
#include "model_d_4_int8.h" 
#include "test_data_kasus_d.h"

// 2. SERTAKAN PUSTAKA TFLM
// =======================================
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
// =======================================
#define NUM_CLASSES 3
#define TF_NUM_OPS 5
#define ARENA_SIZE 12000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
// =======================================
// !!! GANTI NILAI DI BAWAH INI SESUAI OUTPUT DARI SKRIP PYTHON ANDA !!!
const float input_scale = 0.036598969250917435; 
const int input_zero_point = -61;
// Untuk klasifikasi, kita tidak perlu de-kuantisasi output, jadi nilai di bawah ini tidak digunakan.
// const float output_scale = 0.0;
// const int output_zero_point = 0;

/**
 * @brief Menjalankan benchmark pada seluruh data uji.
 */
void runBenchmark() {
    int correct_predictions = 0;
    unsigned long total_time_us = 0;

    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // --- Kuantisasi Manual Input ---
        int8_t input_quantized[FEATURES_PER_SAMPLE_D];
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            float feature_float = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
            input_quantized[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }
        
        int expected_label = test_labels_d[i];

        // Jalankan inferensi dengan input int8
        unsigned long start_us = micros();
        if (!tf.predict(input_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // --- Logika Prediksi untuk Multikelas INT8 ---
        // Cari indeks dari output neuron dengan skor integer tertinggi.
        // Tidak perlu de-kuantisasi untuk mencari kelas mana yang menang.
        int predicted_label = -1;
        int8_t max_score = -128; // Nilai int8 terendah
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (tf.output(j) > max_score) {
                max_score = tf.output(j);
                predicted_label = j;
            }
        }

        // Akumulasi metrik
        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        total_time_us += duration_us;

        // Cetak hasil per baris
        Serial.print(i + 1); Serial.print("\t");
        Serial.print(expected_label); Serial.print("\t");
        Serial.print(predicted_label); Serial.print("\t\t");
        Serial.println(duration_us);
    }

    Serial.println("===========================================");
    Serial.print("Akurasi Test: ");
    Serial.print((float)correct_predictions / TEST_SAMPLES_COUNT_D * 100, 2);
    Serial.println(" %");
    Serial.print("Rata-rata waktu inferensi: ");
    Serial.print((float)total_time_us / TEST_SAMPLES_COUNT_D, 2);
    Serial.println(" us");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("--- Benchmark Model Klasifikasi INT8 (Kasus D) ---");

    tf.setNumInputs(FEATURES_PER_SAMPLE_D);
    tf.setNumOutputs(NUM_CLASSES);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddSoftmax();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_d_model_4_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    runBenchmark();
}

void loop() { }