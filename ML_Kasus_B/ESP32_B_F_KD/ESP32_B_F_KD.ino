// 1. SERTAKAN HEADER MODEL & DATA UJI
#include "model_b_nl_4_float.h"
#include "test_data_kasus_b.h"

// 2. SERTAKAN PUSTAKA TFLM
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL
#define TF_NUM_OPS 6
#define ARENA_SIZE 8000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

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
    tf.resolver.AddReshape();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array
    // Ganti nama variabel ini jika menguji model yang berbeda [cite: 53]
    while (!tf.begin(g_kasus_b_nl_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik untuk pengukuran daya baseline
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus B Non-Linear) ---");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("---------------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus untuk inferensi.
 */
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        // Buat buffer sementara untuk input [cite: 44]
        float input_sample_buffer[FEATURES_PER_SAMPLE_B_NL];
        input_sample_buffer[0] = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL];
        
        float expected_y = test_labels_b_nl[i];

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_time = micros();
        if (!tf.predict(input_sample_buffer).isOk()) { 
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;
        
        // Baca hasil prediksi
        float predicted_y = tf.output();
        
        // Cetak hasil per baris [cite: 48]
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", input_sample_buffer[0], expected_y, predicted_y, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}