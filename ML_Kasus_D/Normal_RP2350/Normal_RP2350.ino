#include <Chirale_TensorFlowLite.h>
#include "model_d_2_float.h"
#include "test_data_kasus_d.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 15360;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void runBenchmark() {
    unsigned long total_latency = 0;
    int correct_predictions = 0;

    Serial.println("Starting benchmark for FLOAT model (Case D)...");
    Serial.println("--------------------------------------------------");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_D; i++) {
        for (int j = 0; j < FEATURES_PER_SAMPLE_D; j++) {
            input->data.f[j] = test_features_d[i * FEATURES_PER_SAMPLE_D + j];
        }

        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        total_latency += latency;

        int predicted_label = 0;
        float max_value = output->data.f[0];
        for (int k = 1; k < 3; k++) {
            if (output->data.f[k] > max_value) {
                max_value = output->data.f[k];
                predicted_label = k;
            }
        }
        
        int expected_label = test_labels_d[i];
        
        if (predicted_label == expected_label) {
            correct_predictions++;
        }
        
        Serial.printf("%.4f\t%d\t%d\t%lu\n", test_features_d[i * FEATURES_PER_SAMPLE_D], expected_label, predicted_label, latency);
    }

    Serial.println("==================================");
    Serial.println("Benchmark finished. Results:");
    
    float accuracy = (float)correct_predictions / TEST_SAMPLES_COUNT_D * 100.0f;
    Serial.printf("Accuracy: %.2f %%\n", accuracy);
    
    unsigned long avg_latency = total_latency / TEST_SAMPLES_COUNT_D;
    Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg_latency);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);

    Serial.println("--- TFLite FLOAT Model Benchmark (Case D, Accuracy) for RP2350/RP2040 ---");
    
    model = tflite::GetModel(g_kasus_d_model_2_float_model_data); 
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

    runBenchmark();
}

void loop() {}