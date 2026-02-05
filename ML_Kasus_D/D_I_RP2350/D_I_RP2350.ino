// 1. SERTAKAN PUSTAKA RP2350 & TFLM
#include <Chirale_TensorFlowLite.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// 2. SERTAKAN HEADER MODEL INT8 & DATA UJI
// Pastikan nama file header di bawah ini sesuai dengan file Kasus D INT8 Anda
#include "model_d_4_int8.h"
#include "test_data_kasus_d.h"

// 3. DEFINISI MODEL & KONFIGURASI
// Variabel global untuk TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Ukuran Tensor Arena, sesuaikan dari referensi ESP32
constexpr int kTensorArenaSize = 12000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const int kNumClasses = 3; // Jumlah kelas output

// 4. PARAMETER KUANTISASI (Salin dari referensi ESP32)
// !!! PASTIKAN SEMUA NILAI INI SESUAI DENGAN MODEL ANDA !!!
const float input_scale = 0.03787701204419136;
const int input_zero_point = -53;
const float output_scale = 0.00390625;
const int output_zero_point = -128;


/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model INT8
    model = tflite::GetModel(g_kasus_d_model_4_int8_model_data);
    
    // Verifikasi versi TFLM
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        while (true);
    }

    // Gunakan AllOpsResolver seperti pada referensi RP2350
    static tflite::AllOpsResolver resolver;

    // Inisialisasi interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Alokasi memori untuk tensor
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed.");
        while (true);
    }
    
    // Dapatkan pointer ke tensor input dan output
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Verifikasi Tipe Data (PENTING untuk Kuantisasi)
    if (input->type != kTfLiteInt8) {
        Serial.println("Tipe tensor input bukan Int8!");
        while(true);
    }
    if (output->type != kTfLiteInt8) {
        Serial.println("Tipe tensor output bukan Int8!");
        while(true);
    }
    
    // Fase Idle selama 5 detik
    delay(5000);
    
    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, LOW);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus D - Quantized) ---");
    Serial.println("Sample\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus.
 */
void loop() {
    // Mengulang benchmark untuk semua data tes
    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        
        // --- 1. Kuantisasi Manual Input ---
        // Mengisi tensor input int8
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            float feature_float = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
            // Terapkan rumus kuantisasi dari referensi ESP32
            input->data.int8[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }

        // --- 2. Menjalankan Inferensi ---
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // --- 3. De-kuantisasi Output ---
        // Memproses output int8 untuk klasifikasi
        int predicted_label = -1;
        float max_prob = -1000.0f; // Gunakan nilai yang sangat kecil

        for (int j = 0; j < kNumClasses; j++) {
            // Ambil output int8
            int8_t output_quantized = output->data.int8[j];
            
            // Ubah ke float (probabilitas) menggunakan rumus de-kuantisasi
            float current_prob = (float)(output_quantized - output_zero_point) * output_scale;

            // Cari probabilitas tertinggi
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = j;
            }
        }
        
        int expected_label = test_labels_d[i];
        
        // --- 4. Mencetak Hasil ---
        // Mencetak hasil per baris
        Serial.printf("%d\t%d\t\t%d\t\t%lu\n", i, expected_label, predicted_label, latency);
    }
    // Loop akan otomatis dimulai lagi
}