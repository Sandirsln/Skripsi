#include <Chirale_TensorFlowLite.h>
#include "model_e_4_float.h"      // g_kasus_e_model_1_float_model_data
#include "test_data_kasus_e.h"    // test_features_e[], test_labels_e[], FEATURES_PER_SAMPLE_E(=600), TEST_SAMPLES_COUNT_E

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// TFLM objects
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// Arena: sesuaikan bila OOM
constexpr int kTensorArenaSize = 30000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Label (urutkan sesuai training)
static const char* kLabels[3] = {"idle", "jog", "walk"};

static void softmax3(const float *x, float *y) {
  float m = x[0];
  if (x[1] > m) m = x[1];
  if (x[2] > m) m = x[2];
  float e0 = expf(x[0] - m);
  float e1 = expf(x[1] - m);
  float e2 = expf(x[2] - m);
  float s = e0 + e1 + e2 + 1e-12f;
  y[0] = e0 / s; y[1] = e1 / s; y[2] = e2 / s;
}

void runBenchmark() {
  unsigned long total_latency = 0;
  int correct = 0;

  Serial.println("Idx\tActual\tPred(Label)\tConf\tTime (us)");

  for (int i = 0; i < TEST_SAMPLES_COUNT_E; i++) {
    // isi input (600 float)
    const int base = i * FEATURES_PER_SAMPLE_E;
    for (int j = 0; j < FEATURES_PER_SAMPLE_E; j++)
      input->data.f[j] = test_features_e[base + j];

    unsigned long t0 = micros();
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }
    unsigned long dt = micros() - t0;

    // ambil output -> probs
    float logits[3] = { output->data.f[0], output->data.f[1], output->data.f[2] };
    float probs[3];
    softmax3(logits, probs); // aman walau model sudah punya Softmax

    // top-1
    int pred = 0;
    float conf = probs[0];
    if (probs[1] > conf) { conf = probs[1]; pred = 1; }
    if (probs[2] > conf) { conf = probs[2]; pred = 2; }

    const int expected = test_labels_e[i];
    if (pred == expected) correct++;
    total_latency += dt;

    Serial.print(i + 1); Serial.print('\t');
    Serial.print(expected); Serial.print('\t');
    Serial.print(pred); Serial.print(" ("); Serial.print(kLabels[pred]); Serial.print(")\t");
    Serial.print(conf, 4); Serial.print('\t');
    Serial.println(dt);
  }

  Serial.println("==================================");
  float acc = (float)correct / TEST_SAMPLES_COUNT_E * 100.0f;
  Serial.print("Akurasi Test: "); Serial.print(acc, 2); Serial.println(" %");
  unsigned long avg = total_latency / TEST_SAMPLES_COUNT_E;
  Serial.print("Rata-rata waktu inferensi: "); Serial.print(avg); Serial.println(" us");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(500);
  Serial.println("--- TFLite FLOAT Model Benchmark (Case E, RP2350) ---");

  model = tflite::GetModel(g_kasus_e_model_4_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!"); while (true) {}
  }

  static tflite::AllOpsResolver resolver; // praktis untuk MLP; ganti ke MicroMutableOpResolver bila ingin minimal
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed."); while (true) {}
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  // sanity: pastikan dimensi input sesuai 600 dan output 3 (opsional, cepat)
  if (input->bytes / sizeof(float) < FEATURES_PER_SAMPLE_E || output->bytes / sizeof(float) < 3) {
    Serial.println("Tensor shape mismatch."); while (true) {}
  }

  runBenchmark();
}

void loop() {}
