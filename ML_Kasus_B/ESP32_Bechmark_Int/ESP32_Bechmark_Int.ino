/**
 * @file main_int8_b_nonlinear.cpp
 * @brief Benchmark model regresi non-linear INT8 (Kasus B) pada ESP32.
 * * !!! LANGKAH WAJIB SEBELUM UPLOAD !!!
 * 1. Ganti file header model (misal: ke model_b_nl_2_int8.h).
 * 2. SALIN & TEMPEL nilai 'scale' dan 'zero_point' dari output skrip Python 
 * (Bagian 8) ke dalam konstanta di bawah ini.
 * 3. Sesuaikan nama variabel model di dalam tf.begin().
 */

// 1. SERTAKAN HEADER MODEL & DATA UJI
// =======================================
#include "model_b_nl_1_int8.h" 
#include "test_data_kasus_b.h"

// 2. SERTAKAN PUSTAKA TFLM
// =======================================
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
// =======================================
#define TF_NUM_OPS 5
#define ARENA_SIZE 8000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
// =======================================
// !!! GANTI NILAI DI BAWAH INI SESUAI OUTPUT DARI SKRIP PYTHON ANDA !!!
const float input_scale = 0.0784313753247261; 
const int input_zero_point = -1;
const float output_scale = 0.23830443620681763;
const int output_zero_point = -122;

/**
 * @brief Menjalankan benchmark pada seluruh data uji.
 */
void runBenchmark() {
    float total_squared_error = 0;
    unsigned long total_time_us = 0;

    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("---------------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        // Ambil satu sampel input (float) dan label yang diharapkan
        float x_float = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL];
        float expected_y = test_labels_b_nl[i];

        // --- Proses Kuantisasi Manual ---
        int8_t x_quantized = (int8_t)(round(x_float / input_scale) + input_zero_point);

        // Jalankan inferensi dengan input int8
        unsigned long start_us = micros();
        if (!tf.predict(&x_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // --- Proses De-kuantisasi Manual Output ---
        int8_t y_quantized = tf.output();
        float predicted_y = (y_quantized - output_zero_point) * output_scale;
        
        float error = expected_y - predicted_y;
        total_squared_error += (error * error);
        
        total_time_us += duration_us;

        // Cetak hasil per baris
        Serial.print(x_float, 4); Serial.print("\t");
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
    Serial.println("--- Benchmark Model INT8 (Kasus B Non-Linear) ---");

    tf.setNumInputs(1);
    tf.setNumOutputs(1);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_b_nl_model_1_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
        
    runBenchmark();
}

/**
 * @brief Fungsi loop utama, tidak digunakan.
 */
void loop() {
    // Kosong.
}