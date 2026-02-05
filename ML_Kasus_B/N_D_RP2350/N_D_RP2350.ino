#include <Chirale_TensorFlowLite.h>
#include "model_b_nl_2_float.h"
#include "test_data_kasus_b.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// Variabel global
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 4096;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Fungsi setup untuk inisialisasi
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH); // Pastikan LED mati di awal

    // Menunggu koneksi serial dari Python script
    delay(1000);

    Serial.println("--- TFLite CONTINUOUS Benchmark with LED Indicator ---");
    
    // Muat model dan siapkan interpreter
    model = tflite::GetModel(g_kasus_b_nl_model_2_float_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        while (true);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed.");
        while (true);
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("Setup complete. Entering 3-second idle period...");
    
    // Periode Idle selama 3 detik
    delay(5000);

    // Aktifkan LED sebagai tanda inferensi dimulai
    Serial.println("Idle period finished. Starting continuous inference loop...");
    Serial.println("--------------------------------------------------");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");
    digitalWrite(LED_BUILTIN, LOW); // Nyalakan LED
}

// Logika benchmark di dalam loop()
void loop() {
    // Loop ini akan terus berulang, menjalankan benchmark lagi dan lagi
    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        // Mengisi data input
        for (int j = 0; j < FEATURES_PER_SAMPLE_B_NL; j++) {
            input->data.f[j] = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL + j];
        }

        // Menjalankan inferensi dan mengukur waktu
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // Membaca hasil
        float predicted_value = output->data.f[0];
        float expected_value = test_labels_b_nl[i];
        
        // Mencetak hasil per baris
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL], expected_value, predicted_value, latency);
    }
}