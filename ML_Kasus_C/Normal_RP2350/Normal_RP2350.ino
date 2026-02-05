#include <Chirale_TensorFlowLite.h>
#include "model_c_3_float.h"
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

// --- Arena Tensor ---
// Alokasikan memori untuk model. Ukuran mungkin perlu disesuaikan.
constexpr int kTensorArenaSize = 10240;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


/**
 * @brief Menjalankan benchmark inferensi pada seluruh dataset tes.
 * * Fungsi ini mengiterasi semua sampel tes, melakukan inferensi,
 * dan mencetak hasilnya ke Serial Monitor dengan format yang
 * dirancang untuk diparsing oleh skrip Python pendamping.
 */
void runBenchmark() {
    unsigned long total_latency = 0;
    int correct_predictions = 0;

    Serial.println("Starting benchmark for FLOAT model (Case C)...");
    
    // --- Header Tabel: Format ini harus cocok dengan yang diharapkan skrip Python ---
    Serial.println("--------------------------------------------------");
    Serial.println("Sample\tActual\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_C; i++) {
        // Salin data fitur dari sampel saat ini ke tensor input model
        for (int j = 0; j < FEATURES_PER_SAMPLE_C; j++) {
            input->data.f[j] = test_features_c[i * FEATURES_PER_SAMPLE_C + j];
        }

        // Ukur waktu inferensi
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        total_latency += latency;

        // Dapatkan hasil prediksi dari tensor output
        float predicted_value = output->data.f[0];
        int predicted_label = (predicted_value > 0.5f) ? 1 : 0;
        
        // Dapatkan label yang sebenarnya (ground truth)
        int expected_label = test_labels_c[i];
        
        // Periksa apakah prediksi benar
        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        
        // --- Mencetak Baris Data Tabel ---
        // PERUBAHAN 1: Mencetak indeks 'i' sebagai kolom "Sample" dan menggunakan
        // tab tunggal '\t' sebagai pemisah agar konsisten.
        Serial.printf("%d\t%d\t%d\t%lu\n", i, expected_label, predicted_label, latency);
    }

    // --- Mencetak Penanda Akhir Tabel ---
    // Skrip Python menggunakan ini untuk berhenti membaca data tabel.
    Serial.println("==================================");
    Serial.println("Benchmark finished. Results:");
    
    // Hitung dan cetak metrik ringkasan
    float accuracy = (float)correct_predictions / TEST_SAMPLES_COUNT_C * 100.0f;
    
    // --- Mencetak Ringkasan Akurasi ---
    // PERUBAHAN 2: Mengubah format string agar cocok persis dengan "Akurasi Test:"
    // yang dicari oleh skrip Python.
    Serial.printf("Akurasi Test: %.2f %%\n", accuracy);
    
    // --- Mencetak Ringkasan Waktu ---
    // Format ini sudah cocok dengan skrip Python.
    unsigned long avg_latency = total_latency / TEST_SAMPLES_COUNT_C;
    Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg_latency);
}

/**
 * @brief Fungsi setup standar Arduino.
 * * Menginisialisasi komunikasi serial dan mempersiapkan model TFLite
 * untuk inferensi sebelum menjalankan benchmark.
 */
void setup() {
    Serial.begin(115220); // Baud rate disesuaikan dengan skrip Python
    while (!Serial); // Tunggu koneksi serial
    delay(1000);

    Serial.println("--- TFLite FLOAT Model Benchmark (Case C, Accuracy) for RP2350/RP2040 ---");
    
    // Muat model dari data header
    model = tflite::GetModel(g_kasus_c_model_3_float_model_data); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        while (true);
    }

    // Siapkan interpreter TFLite
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Alokasikan tensor
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed.");
        while (true);
    }
    
    // Dapatkan pointer ke tensor input dan output
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Jalankan fungsi benchmark utama
    runBenchmark();
}

/**
 * @brief Fungsi loop standar Arduino.
 * * Dibiarkan kosong karena benchmark hanya perlu dijalankan sekali di setup.
 */
void loop() {
    // Tidak ada yang dilakukan di sini
}
