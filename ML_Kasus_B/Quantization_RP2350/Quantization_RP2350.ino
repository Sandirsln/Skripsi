#include <Chirale_TensorFlowLite.h>
#include "model_b_nl_4_int8.h"
#include "test_data_kasus_b.h"

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

float input_scale = 0.0784313753247261;
int8_t input_zero_point = -1;
float output_scale = 0.2810456454753876;
int8_t output_zero_point = -117;

constexpr int kTensorArenaSize = 4096;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void runBenchmark() {
    unsigned long total_latency = 0;
    float total_squared_error = 0.0;

    Serial.println("Starting benchmark for INT8 model (Case B)...");
    Serial.println("--------------------------------------------------");
    Serial.println("Input\tExpected\tPredicted\tTime (us)");
    Serial.println("--------------------------------------------------");

    for (int i = 0; i < TEST_SAMPLES_COUNT_B_NL; i++) {
        for (int j = 0; j < FEATURES_PER_SAMPLE_B_NL; j++) {
            float input_float = test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL + j];
            input->data.int8[j] = (int8_t)((input_float / input_scale) + input_zero_point);
        }

        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long latency = micros() - start_time;
        total_latency += latency;

        int8_t predicted_quantized = output->data.int8[0];
        float predicted_value = (float)(predicted_quantized - output_zero_point) * output_scale;
        
        float expected_value = test_labels_b_nl[i];
        float error = predicted_value - expected_value;
        total_squared_error += error * error;
        
        Serial.printf("%.4f\t%.4f\t\t%.4f\t\t%lu\n", test_features_b_nl[i * FEATURES_PER_SAMPLE_B_NL], expected_value, predicted_value, latency);
    }

    Serial.println("==================================");
    Serial.println("Benchmark finished. Results:");
    
    float mean_squared_error = total_squared_error / TEST_SAMPLES_COUNT_B_NL;
    Serial.printf("Mean Squared Error (MSE): %.4f\n", mean_squared_error);
    
    unsigned long avg_latency = total_latency / TEST_SAMPLES_COUNT_B_NL;
    Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg_latency);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);

    Serial.println("--- TFLite INT8 Model Benchmark (Case B, MSE) for RP2350/RP2040 ---");
    
    model = tflite::GetModel(g_kasus_b_nl_model_4_int8_model_data); 
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
    
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;

    runBenchmark();
}

void loop() {}