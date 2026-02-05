#include <Chirale_TensorFlowLite.h>
#include "model_d_4_int8.h"
#include "test_data_kasus_d.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// --- Variabel Global untuk TensorFlow Lite ---
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// --- Parameter Kuantisasi ---
// Nilai ini akan diperbarui dari model di dalam setup()
float input_scale = 0.03787701204419136f;
int8_t input_zero_point = -53;
float output_scale = 0.00390625f;
int8_t output_zero_point = -128;

// --- Arena Tensor ---
constexpr int kTensorArenaSize = 15360;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

/**
 * @brief Menjalankan benchmark untuk model klasifikasi multi-kelas (INT8).
 * * Logika ini menemukan indeks output dengan nilai tertinggi (argmax)
 * untuk menentukan label prediksi.
 */
void runBenchmark() {
    unsigned long total_latency = 0;
    int correct_predictions = 0;

    Serial.println("Starting benchmark for INT8 model (Case D)...");

    // --- Header Tabel ---
    // PERUBAHAN 1: Menyesuaikan header agar cocok dengan parser Python.
    Serial.println("--------------------------------------------------");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        // Lakukan kuantisasi pada data input
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            float input_float = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
            input->data.int8[j] = (int8_t)((input_float / input_scale) + input_zero_point);
        }

        // Ukur waktu inferensi
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        total_latency += latency;

        // Dapatkan label prediksi (untuk klasifikasi multi-kelas)
        // Cari indeks dari nilai output tertinggi (argmax)
        int predicted_label = 0;
        int8_t max_value = output->data.int8[0];
        for (int k = 1; k < 3; k++) { // Asumsi ada 3 kelas output
            if (output->data.int8[k] > max_value) {
                max_value = output->data.int8[k];
                predicted_label = k;
            }
        }
        
        int expected_label = test_labels_d[i];
        
        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        
        // --- Mencetak Baris Data Tabel ---
        // PERUBAHAN 2: Mencetak indeks 'i' sebagai kolom "Sample".
        Serial.printf("%d\t%d\t%d\t%lu\n", i, expected_label, predicted_label, latency);
    }

    // --- Penanda Akhir Tabel ---
    Serial.println("==================================");
    Serial.println("Benchmark finished. Results:");
    
    float accuracy = (float)correct_predictions / TEST_SAMPLES_COUNT_D * 100.0f;
    
    // --- Mencetak Ringkasan Akurasi ---
    // PERUBAHAN 3: Mengubah "Accuracy:" menjadi "Akurasi Test:".
    Serial.printf("Akurasi Test: %.2f %%\n", accuracy);
    
    unsigned long avg_latency = total_latency / TEST_SAMPLES_COUNT_D;
    Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg_latency);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);

    Serial.println("--- TFLite INT8 Model Benchmark (Case D, Accuracy) for RP2350/RP2040 ---");
    
    model = tflite::GetModel(g_kasus_d_model_4_int8_model_data); 
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
    
    // Ambil parameter kuantisasi langsung dari model
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;

    runBenchmark();
}

void loop() {
    // Dibiarkan kosong
}