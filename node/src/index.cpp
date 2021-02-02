
#include "index.h"

#include "ml.h"
#include "neural_network_context.h"
#include "model_builder.h"
#include "model.h"
#include "compilation.h"
#include "operand.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {

  ML::Initialize(env, exports);
  NeuralNetworkContext::Initialize(env, exports);
  ModelBuilder::Initialize(env, exports);
  Model::Initialize(env, exports);
  Compilation::Initialize(env, exports);
  Operand::Initialize(env, exports);

  return exports;
}

NODE_API_MODULE(addon, Init)
