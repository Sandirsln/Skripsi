// 1. SERTAKAN PUSTAKA RP2350 & TFLM
#include <Chirale_TensorFlowLite.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// 2. SERTAKAN HEADER MODEL INT8 & DATA UJI (Diperbarui untuk Kasus E)
// GANTI nama file header di bawah ini agar sesuai dengan file Kasus E INT8 Anda
#include "model_e_4_int8.h" // <-- GANTI dengan nama model INT8 Kasus E Anda
#include "test_data_kasus_e.h" // <-- GANTI dengan nama data uji Kasus E Anda

// 3. DEFINISI MODEL & KONFIGURASI
// Variabel global untuk TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Ukuran Tensor Arena, sesuaikan dari referensi ESP32/RP2350
constexpr int kTensorArenaSize = 12000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const int kNumClasses = 3; // Jumlah kelas output (Asumsi sama dengan Kasus D)

// 4. PARAMETER KUANTISASI (Diperbarui untuk Kasus E)
// !!! PASTIKAN SEMUA NILAI INI SESUAI DENGAN MODEL KUANTISASI KASUS E ANDA !!!
// Nilai di bawah ini adalah contoh dan HARUS disesuaikan.
const float input_scale = 0.06629331409931183; // <-- GANTI
const int input_zero_point = 2; // <-- GANTI
const float output_scale = 0.00390625; // <-- GANTI (Biasanya sama untuk model yang sama)
const int output_zero_point = -128; // <-- GANTI (Biasanya sama untuk model yang sama)


/**
 * @brief Fungsi setup, dijalankan sekali saat boot.
 */
void setup() {
    Serial.begin(115200);
    
    // Inisialisasi LED, dimulai dalam keadaan mati (HIGH)
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model INT8 (GANTI nama variabel global model)
    // Pastikan nama variabel global model sesuai dengan file header Kasus E Anda
    model = tflite::GetModel(g_kasus_e_model_4_int8_model_data); 
    
    // Verifikasi versi TFLM
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        while (true);
    }

    // Gunakan AllOpsResolver
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

    // Verifikasi Tipe Data
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
    Serial.println("--- Continuous Inference Started (Kasus E - Quantized) ---");
    Serial.println("Sample\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

/**
 * @brief Fungsi loop utama, berjalan terus-menerus.
 */
void loop() {
    // Mengulang benchmark untuk semua data tes
    // GANTI konstanta hitungan data uji
    for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) { 
        
        // --- 1. Kuantisasi Manual Input ---
        // Mengisi tensor input int8
        // GANTI konstanta fitur per sampel
        for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++) { 
            // GANTI nama array fitur
            float feature_float = test_features_e[i * FEATURES_PER_SAMPLE_E + j]; 
            
            // Rumus kuantisasi (sama dengan Kasus D)
            input->data.int8[j] = (int8_t)(round(feature_float / input_scale) + input_zero_point);
        }

        // --- 2. Menjalankan Inferensi ---
        unsigned long start_time = micros();
        TfLiteStatus invoke_status = interpreter->Invoke(); // Ambil status invoke
        unsigned long latency = micros() - start_time;
        
        if (invoke_status != kTfLiteOk) {
            Serial.println("Invoke failed!");
            continue; // Lanjut ke sampel berikutnya
        }
        
        // --- 3. De-kuantisasi Output ---
        // Memproses output int8 untuk klasifikasi
        int predicted_label = -1;
        float max_prob = -1000.0f; 

        for (int j = 0; j < kNumClasses; j++) {
            // Ambil output int8
            int8_t output_quantized = output->data.int8[j];
            
            // Ubah ke float (probabilitas) menggunakan rumus de-kuantisasi (sama dengan Kasus D)
            float current_prob = (float)(output_quantized - output_zero_point) * output_scale;

            // Cari probabilitas tertinggi
            if (current_prob > max_prob) {
                max_prob = current_prob;
                predicted_label = j;
            }
        }
        
        // GANTI nama array label
        int expected_label = test_labels_e[i]; 
        
        // --- 4. Mencetak Hasil ---
        Serial.printf("%d\t%d\t\t%d\t\t%lu\n", i, expected_label, predicted_label, latency);
    }
    // Loop akan otomatis dimulai lagi
}