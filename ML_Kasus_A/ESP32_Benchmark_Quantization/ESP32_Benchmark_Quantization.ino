#include "model_a_4_int8.h" 
#include "test_data_kasus_a.h"

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
const float input_scale = 1.958823561668396; 
const int input_zero_point = -1;
const float output_scale = 3.9176385402679443;
const int output_zero_point = -1;

void runBenchmark() {
    float total_squared_error = 0;
    unsigned long total_time_us = 0;

    Serial.println("Input\tExpected\tPredicted\tSquared Error\tTime (us)");
    Serial.println("-----------------------------------------------------------------");

    // Loop melalui semua sampel data uji dari file header
    for (int i = 0; i < TEST_SAMPLES_COUNT_A; i++) {
        // Ambil satu sampel input (float) dan label yang diharapkan
        float x_float = test_features_a[i * FEATURES_PER_SAMPLE_A];
        float expected_y = test_labels_a[i];

        // --- Proses Kuantisasi Manual ---
        // Ubah input dari float32 ke int8 sesuai scale & zero-point
        int8_t x_quantized = (int8_t)(round(x_float / input_scale) + input_zero_point);

        // Jalankan inferensi dengan input int8
        unsigned long start_us = micros();
        if (!tf.predict(&x_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // Dapatkan output int8 dari model
        int8_t y_quantized = tf.output();

        // --- Proses De-kuantisasi Manual ---
        // Ubah output dari int8 kembali ke float32
        float predicted_y = (y_quantized - output_zero_point) * output_scale;
        
        // Hitung kuadrat eror untuk MSE
        float error = expected_y - predicted_y;
        float squared_error = error * error;

        // Akumulasi metrik
        total_squared_error += squared_error;
        total_time_us += duration_us;

        // Cetak hasil per baris
        Serial.print(x_float, 4); Serial.print("\t");
        Serial.print(expected_y, 4); Serial.print("\t\t");
        Serial.print(predicted_y, 4); Serial.print("\t\t");
        Serial.print(squared_error, 4); Serial.print("\t");
        Serial.println(duration_us);
    }

    // Cetak ringkasan hasil benchmark
    Serial.println("=================================================================");
    Serial.print("Mean Squared Error (MSE): ");
    Serial.println(total_squared_error / TEST_SAMPLES_COUNT_A, 4);
    Serial.print("Rata-rata waktu inferensi: ");
    Serial.print((float)total_time_us / TEST_SAMPLES_COUNT_A, 2);
    Serial.println(" us");
}

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("--- Benchmark Model Regresi INT8 (Kasus A) ---");

    tf.setNumInputs(1);
    tf.setNumOutputs(1);

    // Daftarkan operator yang dibutuhkan
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    tf.resolver.AddReshape();

    // Inisialisasi model dari C array INT8
    // Ganti nama variabel ini jika menguji model yang berbeda
    while (!tf.begin(g_kasus_a_model_4_int8_model_data).isOk()) {
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