#include <Chirale_TensorFlowLite.h>
#include "model_c_4_float.h"
#include "test_data_kasus_c.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// Variabel global untuk TensorFlow Lite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Ukuran Tensor Arena disesuaikan untuk Kasus C
constexpr int kTensorArenaSize = 10240;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Fungsi setup untuk inisialisasi satu kali
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED, dimulai dalam keadaan mati
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model dan menyiapkan interpreter
    model = tflite::GetModel(g_kasus_c_model_4_float_model_data); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // Jika gagal, berhenti di sini
        while (true);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        // Jika gagal, berhenti di sini
        while (true);
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Fase Idle selama 5 detik
    delay(5000);

    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, LOW);
    
    // Mencetak header tabel (opsional, untuk debugging)
    Serial.println("--- Continuous Inference Started (Case C) ---");
    Serial.println("Input_1\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

// Fungsi loop berjalan terus-menerus
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // Mengisi tensor input
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            input->data.f[j] = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
        }

        // Menjalankan inferensi dan mengukur waktu
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // Memproses output untuk klasifikasi
        float predicted_value = output->data.f[0];
        int predicted_label = (predicted_value > 0.5f) ? 1 : 0;
        int expected_label = test_labels_c[i];
        
        // Mencetak hasil per baris (opsional, untuk debugging)
        // Kita hanya cetak fitur pertama sebagai representasi input
        Serial.printf("%.4f\t%d\t\t%d\t\t%lu\n", test_features_c[i * FEATURES_PER_SAMPLE_C], expected_label, predicted_label, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}