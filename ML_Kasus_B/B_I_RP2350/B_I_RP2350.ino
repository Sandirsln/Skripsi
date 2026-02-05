#include <Chirale_TensorFlowLite.h>
// Pastikan nama file header di bawah ini sesuai dengan file kuantisasi Kasus B Anda
#include "model_b_nl_1_int8.h"
#include "test_data_kasus_b.h"

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

// Variabel untuk menyimpan parameter kuantisasi
float input_scale = 0.0784313753247261f;
int8_t input_zero_point = -1;
float output_scale = 0.2810456454753876f;
int8_t output_zero_point = -117;

// Ukuran Tensor Arena
constexpr int kTensorArenaSize = 4096;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Fungsi setup untuk inisialisasi satu kali
void setup() {
    Serial.begin(115200);

    // Inisialisasi LED, dimulai dalam keadaan mati
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    // Memuat model dan menyiapkan interpreter
    // Pastikan nama variabel model di bawah ini sesuai dengan file header Anda
    model = tflite::GetModel(g_kasus_b_nl_model_1_int8_model_data); 
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
    
    // Mengambil parameter kuantisasi dari model
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;
    
    // Fase Idle selama 5 detik
    delay(5000);

    // Menyalakan LED sebagai tanda fase aktif dimulai
    digitalWrite(LED_BUILTIN, LOW);
    
    // Mencetak header tabel (opsional, untuk debugging)
    Serial.println("--- Continuous Quantized Inference Started (Case B) ---");
    Serial.println("Input_1\tExpected\tPredicted\tTime (us)");
    Serial.println("-------------------------------------------------");
}

// Fungsi loop berjalan terus-menerus
void loop() {
    // Mengulang benchmark untuk semua data tes di dalam loop
    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        
        // 1. KUANTISASI: Ubah 2 fitur input float ke int8 sebelum inferensi
        for (int j = 0; j < FEATURES_PER_SAMPLE_B_NL; j++) {
            float input_float = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL + j];
            input->data.int8[j] = (int8_t)(input_float / input_scale + input_zero_point);
        }

        // Menjalankan inferensi dan mengukur waktu
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        
        // Ambil output sebagai int8
        int8_t output_quantized = output->data.int8[0];

        // 2. DE-KUANTISASI: Ubah output int8 kembali ke float
        float predicted_value = (float)(output_quantized - output_zero_point) * output_scale;
        float expected_value = test_labels_b_nl[i];
        
        // Mencetak hasil per baris (opsional, untuk debugging)
        // Hanya cetak fitur pertama sebagai representasi
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL], expected_value, predicted_value, latency);
    }
    // Setelah selesai, loop() akan otomatis dimulai lagi dari awal
}