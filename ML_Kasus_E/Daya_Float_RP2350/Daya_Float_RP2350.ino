// 1. SERTAKAN PUSTAKA RP2350 & TFLM
#include <Chirale_TensorFlowLite.h>
// GANTI header model float dan data uji ke versi Kasus E
#include "model_e_4_float.h" // <-- PASTIKAN NAMA FILE MODEL FLOAT KASUS E ANDA
#include "test_data_kasus_e.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// 2. DEFINISI MODEL & KONFIGURASI
// Variabel global untuk TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Ukuran Tensor Arena
constexpr int kTensorArenaSize = 10240;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Jumlah kelas output (asumsi 3: idle, jog, walk)
const int kNumClasses = 3; 

// Fungsi setup untuk inisialisasi satu kali
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED, dimulai dalam keadaan mati
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model (GANTI nama variabel global model ke Kasus E)
    model = tflite::GetModel(g_kasus_e_model_4_float_model_data); // <-- PASTIKAN NAMA VARIABEL C ARRAY MODEL FLOAT
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        while (true); 
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Alokasi memori
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed.");
        while (true); 
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Verifikasi Tipe Data
    if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
        Serial.println("Tipe tensor bukan Float32. Cek model Anda.");
        while (true);
    }

    // Fase Idle selama 5 detik
    delay(5000);

    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, LOW);
    
    // Mencetak header tabel
    Serial.println("--- Continuous Inference Started (Kasus E - Float) ---");
    Serial.println("Sample\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

// Fungsi loop berjalan terus-menerus
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    // GANTI konstanta hitungan sampel
    for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) { 
        
        // --- 1. Mengisi Tensor Input ---
        // GANTI konstanta fitur per sampel dan nama array fitur
        for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++) { 
            input->data.f[j] = test_features_e[i * FEATURES_PER_SAMPLE_E + j];
        }

        // --- 2. Menjalankan Inferensi ---
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // --- 3. Memproses Output ---
        // Memproses output untuk klasifikasi multi-kelas (mencari nilai tertinggi)
        int predicted_label = 0;
        float max_value = output->data.f[0];
        
        // Mengiterasi berdasarkan jumlah kelas (kNumClasses = 3), BUKAN jumlah fitur.
        for (int j = 1; j < kNumClasses; j++) { 
            if (output->data.f[j] > max_value) {
                max_value = output->data.f[j];
                predicted_label = j;
            }
        }
        
        // GANTI nama array label
        int expected_label = test_labels_e[i]; 
        
        // --- 4. Mencetak Hasil ---
        Serial.printf("%d\t%d\t\t%d\t\t%lu\n", i, expected_label, predicted_label, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}