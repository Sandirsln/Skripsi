#include <Chirale_TensorFlowLite.h>
// Pastikan nama file header di bawah ini sesuai dengan file Kasus D Anda
#include "model_d_4_float.h"
#include "test_data_kasus_d.h"

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

// Ukuran Tensor Arena, sesuaikan jika perlu
constexpr int kTensorArenaSize = 10240;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Fungsi setup untuk inisialisasi satu kali
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED, dimulai dalam keadaan mati
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model dan menyiapkan interpreter
    // Pastikan nama variabel model di bawah ini sesuai dengan file header Anda
    model = tflite::GetModel(g_kasus_d_model_4_float_model_data); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        while (true); // Berhenti jika ada error
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        while (true); // Berhenti jika ada error
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Fase Idle selama 5 detik
    delay(5000);

    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, LOW);
    
    // Mencetak header tabel (opsional, untuk debugging)
    Serial.println("--- Continuous Inference Started (Case D) ---");
    Serial.println("Sample\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

// Fungsi loop berjalan terus-menerus
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // Mengisi tensor input dengan 19 fitur
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            input->data.f[j] = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
        }

        // Menjalankan inferensi dan mengukur waktu
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // Memproses output untuk klasifikasi multi-kelas (mencari nilai tertinggi)
        int predicted_label = 0;
        float max_value = output->data.f[0];
        for (int j = 1; j < FEATURES_PER_SAMPLE_D; j++) {
            if (output->data.f[j] > max_value) {
                max_value = output->data.f[j];
                predicted_label = j;
            }
        }
        
        int expected_label = test_labels_d[i];
        
        // Mencetak hasil per baris (opsional, untuk debugging)
        Serial.printf("%d\t%d\t\t%d\t\t%lu\n", i, expected_label, predicted_label, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}