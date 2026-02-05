/**
 * @file main_float_c_fixed_v2.cpp
 * @brief Benchmark model klasifikasi biner Float32 (Kasus C) pada ESP32 - VERSI FINAL.
 */

// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_c_4_float.h" 
#include "test_data_kasus_c.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define TF_NUM_OPS 6 
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

void benchmarkModel() {
    int correct_predictions = 0;
    unsigned long total_time_us = 0;

    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // --- PERBAIKAN: Buat buffer non-const sementara untuk input ---
        // 1. Buat array sementara di dalam loop
        float input_sample_buffer[FEATURES_PER_SAMPLE_C];
        // 2. Salin data dari array 'const' ke buffer sementara
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            input_sample_buffer[j] = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
        }
        // ----------------------------------------------------------------

        int expected_label = test_labels_c[i];

        unsigned long start_us = micros();
        // --- PERBAIKAN: Gunakan buffer sementara untuk predict() ---
        if (!tf.predict(input_sample_buffer).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        float probability = tf.output();
        int predicted_label = probability > 0.5 ? 1 : 0;

        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        total_time_us += duration_us;

        Serial.print(i + 1); Serial.print("\t");
        Serial.print(expected_label); Serial.print("\t");
        Serial.print(predicted_label); Serial.print("\t\t");
        Serial.println(duration_us);
    }

    Serial.println("===========================================");
    Serial.print("Akurasi Test: ");
    Serial.print((float)correct_predictions / TEST_SAMPLES_COUNT_C * 100, 2);
    Serial.println(" %");
    Serial.print("Rata-rata waktu inferensi: ");
    Serial.print((float)total_time_us / TEST_SAMPLES_COUNT_C, 2);
    Serial.println(" us");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("--- Benchmark Model Klasifikasi Float32 (Kasus C) ---");

    tf.setNumInputs(FEATURES_PER_SAMPLE_C);
    tf.setNumOutputs(1);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddLogistic(); // Menggunakan AddLogistic untuk Sigmoid
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();

    while (!tf.begin(g_kasus_c_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    benchmarkModel();
}

void loop() { }