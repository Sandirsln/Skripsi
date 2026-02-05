#include <Chirale_TensorFlowLite.h>
#include "model_c_3_int8.h"
#include "test_data_kasus_c.h"

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
// Nilai ini akan diambil langsung dari model di dalam setup()
float input_scale = 0.02916759066283703f;
int8_t input_zero_point = -24;
float output_scale = 0.00390625f;
int8_t output_zero_point = -128;

// --- Arena Tensor ---
constexpr int kTensorArenaSize = 10240;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

/**
 * @brief Menjalankan benchmark inferensi pada seluruh dataset tes untuk model INT8.
 * * Fungsi ini melakukan kuantisasi pada data input, menjalankan inferensi,
 * melakukan de-kuantisasi pada output, dan mencetak hasilnya ke Serial Monitor
 * dengan format yang telah ditentukan.
 */
void runBenchmark() {
    unsigned long total_latency = 0;
    int correct_predictions = 0;

    Serial.println("Starting benchmark for INT8 model (Case C)...");

    // --- Header Tabel: Format ini harus cocok dengan yang diharapkan skrip Python ---
    // PERUBAHAN 1: Mengganti "Input" -> "Sample" dan "Expected" -> "Actual"
    Serial.println("--------------------------------------------------");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // Lakukan kuantisasi pada data input (float -> int8)
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            float input_float = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
            input->data.int8[j] = (int8_t)((input_float / input_scale) + input_zero_point);
        }

        // Ukur waktu inferensi
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        total_latency += latency;

        // Lakukan de-kuantisasi pada data output (int8 -> float)
        int8_t predicted_quantized = output->data.int8[0];
        float predicted_value = (float)(predicted_quantized - output_zero_point) * output_scale;
        int predicted_label = (predicted_value > 0.5f) ? 1 : 0;

        int expected_label = test_labels_c[i];
        
        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        
        // --- Mencetak Baris Data Tabel ---
        // PERUBAHAN 2: Mencetak indeks 'i' sebagai kolom "Sample" untuk konsistensi.
        Serial.printf("%d\t%d\t%d\t%lu\n", i, expected_label, predicted_label, latency);
    }

    // --- Mencetak Penanda Akhir Tabel ---
    Serial.println("==================================");
    Serial.println("Benchmark finished. Results:");
    
    float accuracy = (float)correct_predictions / TEST_SAMPLES_COUNT_C * 100.0f;
    
    // --- Mencetak Ringkasan Akurasi ---
    // PERUBAHAN 3: Mengubah "Accuracy:" menjadi "Akurasi Test:" agar cocok dengan parser.
    Serial.printf("Akurasi Test: %.2f %%\n", accuracy);
    
    unsigned long avg_latency = total_latency / TEST_SAMPLES_COUNT_C;
    Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg_latency);
}

/**
 * @brief Fungsi setup standar Arduino.
 * * Menginisialisasi komunikasi serial dan mempersiapkan model TFLite INT8
 * untuk inferensi sebelum menjalankan benchmark.
 */
void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);
    Serial.println("--- TFLite INT8 Model Benchmark (Case C, Accuracy) for RP2350/RP2040 ---");
    
    model = tflite::GetModel(g_kasus_c_model_3_int8_model_data); 
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
    
    // Praktik terbaik: ambil parameter kuantisasi langsung dari tensor model
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;
    
    runBenchmark();
}

void loop() {
    // Dibiarkan kosong
}
