#include "model_e_4_int8.h"      
#include "test_data_kasus_e.h"

#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define NUM_CLASSES 3
#define TF_NUM_OPS 6
#ifndef ARENA_SIZE
  #define ARENA_SIZE 20000
#endif

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// ==== KUANTISASI INPUT (ISI DARI METADATA MODEL INT8) ====
const float input_scale = 0.06629331409931183f;   // <-- GANTI
const int   input_zero_point = 2;  // <-- GANTI

static inline void quantize_input(const float *src, int8_t *dst, int len, float scale, int zp){
  for(int i=0;i<len;i++){
    const float q = roundf(src[i] / scale) + zp;
    int v = (int)q;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    dst[i] = (int8_t)v;
  }
}

void runBenchmark(){
  Serial.println("Sample\tActual\tPredicted\tTime (us)");
  Serial.println("-------------------------------------------");

  int correct = 0;
  unsigned long tsum = 0;

  int8_t xq[FEATURES_PER_SAMPLE_E];

  for (int i=0; i<TEST_SAMPLES_COUNT_E; i++){
    const int base = i * FEATURES_PER_SAMPLE_E;
    quantize_input(&test_features_e[base], xq, FEATURES_PER_SAMPLE_E, input_scale, input_zero_point);

    const int expected = test_labels_e[i];

    unsigned long t0 = micros();
    if (!tf.predict(xq).isOk()){
      Serial.println(tf.exception.toString());
      return;
    }
    unsigned long dt = micros() - t0;

    int pred = 0; int8_t best = tf.output(0);
    for (int j=1;j<NUM_CLASSES;j++){
      int8_t val = tf.output(j);
      if (val > best){ best = val; pred = j; }
    }

    if (pred == expected) correct++;
    tsum += dt;

    Serial.print(i+1); Serial.print('\t');
    Serial.print(expected); Serial.print('\t');
    Serial.print(pred); Serial.print("\t\t");
    Serial.println(dt);
  }

  Serial.println("===========================================");
  Serial.print("Akurasi Test: ");
  Serial.print((float)correct / TEST_SAMPLES_COUNT_E * 100.0f, 2);
  Serial.println(" %");
  Serial.print("Rata-rata waktu inferensi: ");
  Serial.print((float)tsum / TEST_SAMPLES_COUNT_E, 2);
  Serial.println(" us");
}

void setup(){
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- Benchmark Model Klasifikasi INT8 (Kasus E) ---");

  tf.setNumInputs(FEATURES_PER_SAMPLE_E);
  tf.setNumOutputs(NUM_CLASSES);

  tf.resolver.AddFullyConnected();
  tf.resolver.AddRelu();
  tf.resolver.AddReshape();
  tf.resolver.AddSoftmax();
  tf.resolver.AddQuantize();
  tf.resolver.AddDequantize();

  // GANTI simbol model sesuai header (contoh nama: g_kasus_e_model_1_int8_model_data)
  while (!tf.begin(g_kasus_e_model_4_int8_model_data).isOk()){
    Serial.println(tf.exception.toString());
    delay(800);
  }

  runBenchmark();
}

void loop(){}
