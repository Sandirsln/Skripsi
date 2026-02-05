// 1. SERTAKAN HEADER MODEL & DATA UJI
// =======================================
#include "model_c_4_int8.h" 
#include "test_data_kasus_c.h"

// 2. SERTAKAN PUSTAKA TFLM
// =======================================
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
// =======================================
#define TF_NUM_OPS 5
#define ARENA_SIZE 12000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
// =======================================
// !!! GANTI NILAI DI BAWAH INI SESUAI OUTPUT DARI SKRIP PYTHON ANDA !!!
const float input_scale = 0.025099536404013634; 
const int input_zero_point = 2;
const float output_scale = 0.00390625;
const int output_zero_point = -128;

/**
 * @brief Menjalankan benchmark pada seluruh data uji.
 */
void benchmarkModel() {
    int correct_predictions = 0;
    unsigned long total_time_us = 0;

    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // --- Kuantisasi Manual Input (untuk 19 fitur) ---
        int8_t input_quantized[FEATURES_PER_SAMPLE_C];
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            float feature_float = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
            input_quantized[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }
        
        int expected_label = test_labels_c[i];

        // Jalankan inferensi dengan input int8
        unsigned long start_us = micros();
        if (!tf.predict(input_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // --- De-kuantisasi Manual Output ---
        int8_t output_quantized = tf.output();
        float probability = (float)(output_quantized - output_zero_point) * output_scale;
        int predicted_label = probability > 0.5 ? 1 : 0;

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
    Serial.print((float)correct_predictions / TEST_SAMPLES_COUNT_C * 100, 2);
    Serial.println(" %");
    Serial.print("Rata-rata waktu inferensi: ");
    Serial.print((float)total_time_us / TEST_SAMPLES_COUNT_C, 2);
    Serial.println(" us");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("--- Benchmark Model Klasifikasi INT8 (Kasus C) ---");

    tf.setNumInputs(FEATURES_PER_SAMPLE_C);
    tf.setNumOutputs(1);

    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddLogistic(); // Untuk INT8, Sigmoid biasanya disebut Logistic
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_c_model_4_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    benchmarkModel();
}

void loop() { }