/**
 * @file main_float_b_nonlinear.cpp
 * @brief Benchmark model regresi non-linear Float32 (Kasus B) pada ESP32.
 * * UNTUK MENGUJI MODEL LAIN:
 * 1. Ganti file header model (misal: ke model_b_nl_2_float.h).
 * 2. Sesuaikan nama variabel model di dalam tf.begin().
 */

// 1. SERTAKAN HEADER MODEL & DATA UJI
// =======================================
// Ganti file ini untuk menguji model lain
#include "model_b_nl_4_float.h" 
#include "test_data_kasus_b.h"

// 2. SERTAKAN PUSTAKA TFLM
// =======================================
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
// =======================================
#define TF_NUM_OPS 6 
#define ARENA_SIZE 8000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

/**
 * @brief Menjalankan benchmark pada seluruh data uji.
 */
void benchmarkModel() {
    float total_squared_error = 0;
    unsigned long total_time_us = 0;

    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("---------------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        // --- PERBAIKAN: Buat buffer non-const sementara untuk input ---
        float input_sample_buffer[FEATURES_PER_SAMPLE_B_NL];
        input_sample_buffer[0] = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL];
        
        float expected_y = test_labels_b_nl[i];

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_us = micros();
        if (!tf.predict(input_sample_buffer).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        float predicted_y = tf.output();
        
        float error = expected_y - predicted_y;
        total_squared_error += (error * error);

        total_time_us += duration_us;

        // Cetak hasil per baris
        Serial.print(input_sample_buffer[0], 4); Serial.print("\t");
        Serial.print(expected_y, 4); Serial.print("\t\t");
        Serial.print(predicted_y, 4); Serial.print("\t\t");
        Serial.println(duration_us);
    }

    Serial.println("=========================================================");
    Serial.print("Rata-rata Mean Squared Error (MSE): ");
    Serial.println(total_squared_error / TEST_SAMPLES_COUNT_B_NL, 6);
    Serial.print("Rata-rata waktu inferensi: ");
    Serial.print((float)total_time_us / TEST_SAMPLES_COUNT_B_NL, 2);
    Serial.println(" us");
}

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("--- Benchmark Model Float32 (Kasus B Non-Linear) ---");

    tf.setNumInputs(1); // Kembali ke 1 input
    tf.setNumOutputs(1);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();

    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_b_nl_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
        
    benchmarkModel();
}

/**
 * @brief Fungsi loop utama, tidak digunakan.
 */
void loop() {
    // Kosong.
}