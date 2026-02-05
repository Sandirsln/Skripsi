// 1. SERTAKAN HEADER MODEL & DATA UJI (DIUBAH KE KASUS E)
#include "model_e_4_int8.h" // <-- GANTI: Model Integer Kasus E Anda
#include "test_data_kasus_e.h" // <-- GANTI: Data Uji Kasus E Anda

// 2. SERTAKAN PUSTAKA TFLM (TETAP)
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// 3. KONFIGURASI MODEL (TETAP)
#define NUM_CLASSES 3
#define TF_NUM_OPS 5
#define ARENA_SIZE 12000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// 4. PARAMETER KUANTISASI (WAJIB DIUBAH KE NILAI KASUS E ANDA!)
// !!! PASTIKAN SEMUA NILAI INI SESUAI DENGAN MODEL KUANTISASI KASUS E ANDA !!!
const float input_scale = 0.06629331409931183; // <-- GANTI DENGAN NILAI AKTUAL KASUS E
const int input_zero_point = 2; // <-- GANTI DENGAN NILAI AKTUAL KASUS E
const float output_scale = 0.00390625; // <-- GANTI DENGAN NILAI AKTUAL KASUS E
const int output_zero_point = -128; // <-- GANTI DENGAN NILAI AKTUAL KASUS E

/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED, dimulai dalam keadaan mati (LOW/ACTIVE HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Konfigurasi input/output model (DIUBAH KE KONSTANTA KASUS E)
    tf.setNumInputs(FEATURES_PER_SAMPLE_E); // <-- DIGANTI
    tf.setNumOutputs(NUM_CLASSES);

    // Daftarkan operator yang dibutuhkan (TETAP)
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddReshape();
    tf.resolver.AddSoftmax();
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();
    
    // Inisialisasi model dari C array INT8 (DIUBAH KE VARIABEL KASUS E)
    // Pastikan nama variabel C array model sesuai dengan file header Anda
    while (!tf.begin(g_kasus_e_model_4_int8_model_data).isOk()) { // <-- DIGANTI
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    
    // Fase Idle selama 5 detik
    delay(5000);

    // Menyalakan LED sebagai tanda fase aktif
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus E - Quantized with Dequant) ---");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("-----------------------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus.
 */
void loop() {
    // Mengulang benchmark untuk semua data tes (DIUBAH KE KONSTANTA KASUS E)
    for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) { // <-- DIGANTI
        
        // --- Kuantisasi Manual Input ---
        int8_t input_quantized[FEATURES_PER_SAMPLE_E]; // <-- DIGANTI

        // Mengisi dan mengkuantisasi input
        for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++) { // <-- DIGANTI
            // Menggunakan array fitur Kasus E
            float feature_float = test_features_e[i * FEATURES_PER_SAMPLE_E + j]; // <-- DIGANTI
            
            // Terapkan rumus kuantisasi (sama dengan Kasus D)
            input_quantized[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }
        
        // Menggunakan array label Kasus E
        int expected_label = test_labels_e[i]; // <-- DIGANTI

        // Jalankan inferensi dan ukur waktu
        unsigned long start_time = micros();
        if (!tf.predict(input_quantized).isOk()) {
            Serial.println(tf.exception.toString());
            continue;
        }
        unsigned long latency = micros() - start_time;

        // --- De-kuantisasi Output lalu Cari Probabilitas Tertinggi ---
        int predicted_label = -1;
        float max_prob = -1.0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            // Ambil output int8
            int8_t output_quantized = tf.output(j);

            // Ubah ke float (probabilitas) menggunakan rumus de-kuantisasi
            float current_prob = (float)(output_quantized - output_zero_point) * output_scale;

            // Cari probabilitas tertinggi
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = j;
            }
        }
        
        // Cetak hasil per baris
        Serial.printf("%d\t%d\t%d\t\t%lu\n", i + 1, expected_label, predicted_label, latency);
    }
}