/**
 * @file main_float_d.cpp
 * @brief Benchmark model klasifikasi multikelas Float32 (Kasus D) pada ESP32.
 * * UNTUK MENGUJI MODEL LAIN:
 * 1. Ganti file header model (misal: ke model_d_2_float.h).
 * 2. Sesuaikan nama variabel model di dalam tf.begin().
 */

// 1. SERTAKAN HEADER MODEL & DATA UJI
// =======================================
#include "model_d_1_float.h" 
#include "test_data_kasus_d.h"

// 2. SERTAKAN PUSTAKA TFLM
// =======================================
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
// =======================================
#define NUM_CLASSES 3
#define TF_NUM_OPS 6 
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * @brief Menjalankan benchmark pada seluruh data uji.
 */
void benchmarkModel() {
    int correct_predictions = 0;
    unsigned long total_time_us = 0;

    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // Buat buffer non-const sementara untuk input
        float input_sample_buffer[FEATURES_PER_SAMPLE_D];
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            input_sample_buffer[j] = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
        }
        
        int expected_label = test_labels_d[i];

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_us = micros();
        if (!tf.predict(input_sample_buffer).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // --- Logika Prediksi untuk Multikelas ---
        // Cari indeks dari output neuron dengan probabilitas tertinggi
        int predicted_label = -1;
        float max_prob = -1.0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (tf.output(j) > max_prob) {
                max_prob = tf.output(j);
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
    Serial.println("--- Benchmark Model Klasifikasi Float32 (Kasus D) ---");

    tf.setNumInputs(FEATURES_PER_SAMPLE_D);
    tf.setNumOutputs(NUM_CLASSES);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddSoftmax(); // Menggunakan AddSoftmax untuk output
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();

    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_d_model_1_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    benchmarkModel();
}

void loop() { }