// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_c_4_int8.h"
#include "test_data_kasus_c.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define TF_NUM_OPS 5
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI
const float input_scale = 0.025099536404013634;
const int input_zero_point = 2;
const float output_scale = 0.00390625;
const int output_zero_point = -128;

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Konfigurasi input/output model
    tf.setNumInputs(FEATURES_PER_SAMPLE_C);
    tf.setNumOutputs(1);

    // Daftarkan operator yang dibutuhkan
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddLogistic(); // Untuk INT8, Sigmoid biasanya disebut Logistic
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array INT8
    while (!tf.begin(g_kasus_c_model_4_int8_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik untuk pengukuran daya baseline
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus C - Quantized) ---");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus untuk inferensi.
 */
void loop() {
    // Mengulang untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // --- Kuantisasi Manual Input (untuk beberapa fitur) ---
        int8_t input_quantized[FEATURES_PER_SAMPLE_C];
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            float feature_float = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
            input_quantized[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }
        
        int expected_label = test_labels_c[i];

        // Jalankan inferensi dengan input int8 dan ukur waktunya
        unsigned long start_time = micros();
        if (!tf.predict(input_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // --- De-kuantisasi Manual Output ---
        int8_t output_quantized = tf.output();
        float probability = (float)(output_quantized - output_zero_point) * output_scale;
        int predicted_label = probability > 0.5 ? 1 : 0;
        
        // Cetak hasil per baris
        Serial.printf("%d\t%d\t%d\t\t%lu\n", i + 1, expected_label, predicted_label, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}