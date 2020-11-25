#include "model_builder.h"

#include <iostream>
#include <vector>

#include "DescriptorDecoder.h"
#include "model.h"
#include "operand.h"
#include "ops/constant.h"
#include "ops/conv2d.h"
#include "ops/input.h"
#include "ops/pool2d.h"

Napi::FunctionReference ModelBuilder::constructor;

WNNNamedOperand GetNamedOperand(Napi::Object item, std::string& name) {
  WNNNamedOperand named_operand;
  Napi::Object obj = item;
  Napi::Array property_names = item.GetPropertyNames();
  if (property_names.Length() == 0) {
    // unnamed operand like '{c}'
  } else if (property_names.Length() == 1) {
    uint32_t index = 0;
    name = property_names.Get(index).As<Napi::String>().Utf8Value();
    obj = item.Get(name).As<Napi::Object>();
  } else {
    // assert(0);
  }
  if (!obj.InstanceOf(Operand::constructor.Value())) {
    std::cout << "Expected 'Operand' for 'NamedOperand'" << std::endl;
    return named_operand;
  }
  named_operand.operand = Napi::ObjectWrap<Operand>::Unwrap(obj)->GetOperand();
  return named_operand;
}

ModelBuilder::ModelBuilder(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<ModelBuilder>(info) {
  DawnProcTable backendProcs = dawn_native::GetProcs();
  dawnProcSetProcs(&backendProcs);
  dawn_native::NeuralNetworkContext context;
  model_builder_ =  context.CreateModelBuilder();
}

ModelBuilder::~ModelBuilder() {
  wnnModelBuilderRelease(model_builder_);
}

Napi::Value ModelBuilder::Constant(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Constant>(info);
  node->SetOutput(wnnModelBuilderConstant(model_builder_,
                                          node->GetOperandDescriptor(),
                                          node->GetValue(), node->GetSize()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Input(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Input>(info);
  node->SetOutput(wnnModelBuilderInput(model_builder_, node->GetName().c_str(),
                                       node->GetOperandDescriptor()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Add(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(wnnModelBuilderAdd(model_builder_, node->GetInputs()[0],
                                     node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Mul(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(wnnModelBuilderMul(model_builder_, node->GetInputs()[0],
                                     node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::MatMul(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(wnnModelBuilderMatmul(model_builder_, node->GetInputs()[0],
                                        node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Conv2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Conv2d>(info);
  node->SetOutput(wnnModelBuilderConv2d(model_builder_, node->GetInputs()[0],
                                        node->GetInputs()[1],
                                        node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::MaxPool2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Pool2d>(info);
  node->SetOutput(wnnModelBuilderMaxPool2d(model_builder_, node->GetInputs()[0],
                                           node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::AveragePool2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Pool2d>(info);
  node->SetOutput(wnnModelBuilderAveragePool2d(
      model_builder_, node->GetInputs()[0], node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Relu(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(wnnModelBuilderRelu(model_builder_, node->GetInputs()[0]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::CreateModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Object model = Model::constructor.New({});

  std::vector<WNNNamedOperand> operands;
  uint32_t length = 1;
  if (info[0].IsArray()) {
    Napi::Array array = info[0].As<Napi::Array>();
    length = array.Length();
    for (size_t i = 0; i < length; ++i) {
      Napi::Object item = array.Get(i).As<Napi::Object>();
      std::string name;
      auto named_operand = GetNamedOperand(item, name);
      named_operand.name = name.data();
      operands.push_back(named_operand);
    }
  } else if (info[0].IsObject()) {
    Napi::Object item = info[0].As<Napi::Object>();
    std::string name;
    auto named_operand = GetNamedOperand(item, name);
    named_operand.name = name.data();
    operands.push_back(named_operand);
  }
  Model* unwrapped = Napi::ObjectWrap<Model>::Unwrap(model);
  unwrapped->SetModel(wnnModelBuilderCreateModel(model_builder_,
                                              operands.data(),
                                              length));

  return model;
}

Napi::Object ModelBuilder::Initialize(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  Napi::Function func = DefineClass(
      env, "ModelBuilder",
      {InstanceMethod("constant", &ModelBuilder::Constant, napi_enumerable),
       InstanceMethod("input", &ModelBuilder::Input, napi_enumerable),
       InstanceMethod("add", &ModelBuilder::Add, napi_enumerable),
       InstanceMethod("mul", &ModelBuilder::Mul, napi_enumerable),
       InstanceMethod("matmul", &ModelBuilder::MatMul, napi_enumerable),
       InstanceMethod("conv2d", &ModelBuilder::Conv2d, napi_enumerable),
       InstanceMethod("maxPool2d", &ModelBuilder::MaxPool2d, napi_enumerable),
       InstanceMethod("averagePool2d", &ModelBuilder::AveragePool2d,
                      napi_enumerable),
       InstanceMethod("relu", &ModelBuilder::Relu, napi_enumerable),
       InstanceMethod("createModel", &ModelBuilder::CreateModel,
                      napi_enumerable)});
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("ModelBuilder", func);
  return exports;
}
