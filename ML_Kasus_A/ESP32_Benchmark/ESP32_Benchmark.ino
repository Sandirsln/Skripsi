#include "model_a_4_float.h" 
#include "test_data_kasus_a.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// Jumlah operator mungkin perlu disesuaikan, tapi 6 adalah awal yang baik.
#define TF_NUM_OPS 6 
// Ukuran arena tensor, mungkin perlu diperbesar untuk model yang lebih kompleks.
#define ARENA_SIZE 8000 

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

void benchmarkModel() {
    float total_squared_error = 0;
    unsigned long total_time_us = 0;

    Serial.println("Input\tExpected\tPredicted\tSquared Error\tTime (us)");
    Serial.println("-----------------------------------------------------------------");

    // Loop melalui semua sampel data uji dari file header
    for (int i = 0; i < TEST_SAMPLES_COUNT_A; i++) {
        // Ambil satu sampel input dan label yang diharapkan
        float x = test_features_a[i * FEATURES_PER_SAMPLE_A];
        float expected_y = test_labels_a[i];

        // Jalankan inferensi dan ukur waktunya
        unsigned long start_us = micros();
        if (!tf.predict(&x).isOk()) {
            Serial.println(tf.exception.toString());
            return;
        }
        unsigned long duration_us = micros() - start_us;
        
        // Dapatkan hasil prediksi
        float predicted_y = tf.output();
        float error = expected_y - predicted_y;
        float squared_error = error * error; // Hitung kuadrat eror untuk MSE

        // Akumulasi metrik
        total_squared_error += squared_error;
        total_time_us += duration_us;

        // Cetak hasil per baris
        Serial.print(x, 4); Serial.print("\t");
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
    Serial.println("--- Benchmark Model Regresi Float32 (Kasus A) ---");

    // Konfigurasi input/output model
    tf.setNumInputs(1);
    tf.setNumOutputs(1);

    // Daftarkan semua operator yang mungkin digunakan oleh model
    // Jika ada error "Op not registered", tambahkan operator yang kurang di sini
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    // Quantize & Dequantize sering kali tetap ada sebagai op pass-through
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();

    // Inisialisasi model dari C array
    // Pastikan nama variabel ini sesuai dengan yang ada di file header model
    while (!tf.begin(g_kasus_a_model_4_float_model_data).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
        
    // Jalankan fungsi benchmark
    benchmarkModel();
}

/**
 * @brief Fungsi loop utama, tidak digunakan dalam kasus ini.
 */
void loop() {
    // Kosong, karena benchmark hanya perlu dijalankan sekali.
}