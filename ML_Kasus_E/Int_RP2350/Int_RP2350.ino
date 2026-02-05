#include <Chirale_TensorFlowLite.h>
#include "model_e_4_int8.h"       // g_kasus_e_model_1_int8_model_data
#include "test_data_kasus_e.h"    // test_features_e[], test_labels_e[], FEATURES_PER_SAMPLE_E, TEST_SAMPLES_COUNT_E

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

// Akan ditimpa dari metadata tensor pada setup()
float  input_scale  = 0.06629331409931183f;
int8_t input_zero_point = 2;
float  output_scale = 0.00390625f;
int8_t output_zero_point = -128;

constexpr int kTensorArenaSize = 15360;   // naikkan jika AllocateTensors gagal (mis. 20000)
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

void runBenchmark() {
  unsigned long total_latency = 0;
  int correct = 0;

  Serial.println("Starting benchmark for INT8 model (Case E)...");
  Serial.println("--------------------------------------------------");
  Serial.println("Sample\tActual\tPredicted\tTime (us)");
  Serial.println("--------------------------------------------------");

  for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) {
    // quantize input window (600 fitur)
    for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++) {
      float x = test_features_e[i * FEATURES_PER_SAMPLE_E + j];
      input->data.int8[j] = (int8_t)((x / input_scale) + input_zero_point);  // selaras referensi Kasus D
    }

    unsigned long t0 = micros();
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }
    unsigned long dt = micros() - t0;
    total_latency += dt;

    // argmax pada output INT8 (3 kelas)
    int pred = 0;
    int8_t best = output->data.int8[0];
    for (int k = 1; k < 3; k++) {
      int8_t v = output->data.int8[k];
      if (v > best) { best = v; pred = k; }
    }

    int expected = test_labels_e[i];
    if (pred == expected) correct++;

    Serial.printf("%d\t%d\t%d\t%lu\n", i, expected, pred, dt);
  }

  Serial.println("==================================");
  Serial.println("Benchmark finished. Results:");
  float acc = (float)correct / TEST_SAMPLES_COUNT_E * 100.0f;
  Serial.printf("Akurasi Test: %.2f %%\n", acc);
  unsigned long avg = total_latency / TEST_SAMPLES_COUNT_E;
  Serial.printf("Rata-rata waktu inferensi: %lu us\n", avg);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(1000);

  Serial.println("--- TFLite INT8 Model Benchmark (Case E, RP2350/RP2040) ---");

  model = tflite::GetModel(g_kasus_e_model_4_int8_model_data);   // ganti simbol jika berbeda
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (true) {}
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    while (true) {}
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  // ambil quant params dari tensor
  input_scale        = input->params.scale;
  input_zero_point   = (int8_t)input->params.zero_point;
  output_scale       = output->params.scale;
  output_zero_point  = (int8_t)output->params.zero_point;

  runBenchmark();
}

void loop() {}
