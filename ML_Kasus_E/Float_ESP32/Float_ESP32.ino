
#include "model_e_4_float.h"
#include "test_data_kasus_e.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define NUM_CLASSES 3
#define TF_NUM_OPS 6
#ifndef ARENA_SIZE
  #define ARENA_SIZE 20000
#endif

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

static inline const char* labelName(int i){
  switch(i){case 0:return "idle";case 1:return "jog";case 2:return "walk";default:return "?";}
}
static inline int argmax(const float *x, int n, float &mx){
  int k=0; mx=x[0];
  for(int i=1;i<n;i++) if(x[i]>mx){mx=x[i];k=i;}
  return k;
}
static inline void getInput(int idx, float *buf){
  int base = idx * FEATURES_PER_SAMPLE_E;
  for(int j=0;j<FEATURES_PER_SAMPLE_E;j++) buf[j]=test_features_e[base+j];
}
static inline bool infer(float *in, float *out, unsigned long &dt){
  unsigned long t0=micros();
  if(!tf.predict(in).isOk()){ Serial.println(tf.exception.toString()); return false; }
  dt = micros()-t0;
  for(int i=0;i<NUM_CLASSES;i++) out[i]=tf.output(i);
  return true;
}

void benchmark(){
  Serial.println("Idx\tActual\tPred(Label)\tConf\tTime (us)");
  int correct=0; unsigned long tsum=0;
  float x[FEATURES_PER_SAMPLE_E], y[NUM_CLASSES], mx;

  for(int i=0;i<TEST_SAMPLES_COUNT_E;i++){
    getInput(i,x);
    unsigned long dt;
    if(!infer(x,y,dt)) return;
    int pred = argmax(y,NUM_CLASSES,mx);
    if(pred==test_labels_e[i]) correct++;
    tsum += dt;

    Serial.print(i+1);Serial.print('\t');
    Serial.print(test_labels_e[i]);Serial.print('\t');
    Serial.print(pred);Serial.print(" (");Serial.print(labelName(pred));Serial.print(")\t");
    Serial.print(mx,4);Serial.print('\t');
    Serial.println(dt);
  }

  Serial.println("==============================================");
  Serial.print("Akurasi Test: "); Serial.print((float)correct/TEST_SAMPLES_COUNT_E*100.0f,2); Serial.println(" %");
  Serial.print("Rata-rata waktu inferensi: "); Serial.print((float)tsum/TEST_SAMPLES_COUNT_E,2); Serial.println(" us");
}

void setup(){
  Serial.begin(115200);
  delay(800);
  Serial.println("\n--- Kasus E Float32: 100x6 -> 3 kelas ---");

  tf.setNumInputs(FEATURES_PER_SAMPLE_E);
  tf.setNumOutputs(NUM_CLASSES);

  tf.resolver.AddFullyConnected();
  tf.resolver.AddRelu();
  tf.resolver.AddReshape();
  tf.resolver.AddSoftmax();
  tf.resolver.AddQuantize();
  tf.resolver.AddDequantize();

  while(!tf.begin(g_kasus_e_model_4_float_model_data).isOk()){
    Serial.println(tf.exception.toString());
    delay(800);
  }
  benchmark();
}

void loop(){}
