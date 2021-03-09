
#include "Index.h"

#include "Ml.h"
#include "NeuralNetworkContext.h"
#include "ModelBuilder.h"
#include "Model.h"
#include "Compilation.h"
#include "Operand.h"

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
