// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_a_2_int8.h"
#include "test_data_kasus_a.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define TF_NUM_OPS 5
#define ARENA_SIZE 8000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
const float input_scale = 1.958823561668396;
const int input_zero_point = -1;
const float output_scale = 3.9178669452667236;
const int output_zero_point = -1;

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Konfigurasi input/output model
    tf.setNumInputs(1);
    tf.setNumOutputs(1);

    // Daftarkan operator yang dibutuhkan
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    tf.resolver.AddReshape();

    // Inisialisasi model dari C array INT8
    while (!tf.begin(g_kasus_a_model_2_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik untuk pengukuran daya baseline
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus A - Quantized) ---");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("---------------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus untuk inferensi.
 */
void loop() {
    // Mengulang untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_A; i++) {
        // Ambil satu sampel input (float) dan label yang diharapkan
        float x_float = test_features_a[i * FEATURES_PER_SAMPLE_A];
        float expected_y = test_labels_a[i];

        // --- Proses Kuantisasi Manual ---
        int8_t x_quantized = (int8_t)(round(x_float / input_scale) + input_zero_point);
        
        // Jalankan inferensi dan ukur waktunya
        unsigned long start_time = micros();
        if (!tf.predict(&x_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // --- Proses De-kuantisasi Manual ---
        int8_t y_quantized = tf.output();
        float predicted_y = (y_quantized - output_zero_point) * output_scale;
        
        // Cetak hasil per baris
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", x_float, expected_y, predicted_y, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}