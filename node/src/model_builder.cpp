#include "model_builder.h"

#include <iostream>
#include <vector>

#include "model.h"
#include "neural_network_context.h"
#include "operand.h"
#include "ops/constant.h"
#include "ops/conv2d.h"
#include "ops/input.h"
#include "ops/pool2d.h"
#include "ops/reshape.h"
#include "ops/transpose.h"

Napi::FunctionReference ModelBuilder::constructor;

WNNNamedOperands GetNamedOperands(Napi::Object obj) {
  WNNNamedOperands named_operands = dawn_native::CreateNamedOperands();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    Napi::Object item = obj.Get(name).As<Napi::Object>();
    if (!item.InstanceOf(Operand::constructor.Value())) {
      std::cout << "Expected 'Operand' for 'NamedOperand'" << std::endl;
      return named_operands;
    }
    WNNOperand operand = Napi::ObjectWrap<Operand>::Unwrap(item)->GetOperand();
    wnnNamedOperandsSet(named_operands, name.data(), operand);
  }
  return named_operands;
}

ModelBuilder::ModelBuilder(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<ModelBuilder>(info) {
  Napi::Object context = info[0].As<Napi::Object>();
  NeuralNetworkContext *unwrapped =
      Napi::ObjectWrap<NeuralNetworkContext>::Unwrap(context);
  model_builder_ =
      wnnNeuralNetworkContextCreateModelBuilder(unwrapped->GetContext());
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

Napi::Value ModelBuilder::Reshape(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Reshape>(info);
  node->SetOutput(wnnModelBuilderReshape(model_builder_, node->GetInputs()[0],
                                         node->GetNewShape().data(),
                                         node->GetNewShape().size()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Softmax(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(wnnModelBuilderSoftmax(model_builder_, node->GetInputs()[0]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Transpose(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Transpose>(info);
  node->SetOutput(wnnModelBuilderTranspose(model_builder_, node->GetInputs()[0],
                                           node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::CreateModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  WNNNamedOperands named_operands =
      GetNamedOperands(info[0].As<Napi::Object>());
  std::vector<napi_value> args = {info[0].As<Napi::Value>()};
  Napi::Object model = Model::constructor.New(args);
  Model* unwrapped = Napi::ObjectWrap<Model>::Unwrap(model);
  unwrapped->SetModel(
      wnnModelBuilderCreateModel(model_builder_, named_operands));

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
       InstanceMethod("reshape", &ModelBuilder::Reshape, napi_enumerable),
       InstanceMethod("softmax", &ModelBuilder::Softmax, napi_enumerable),
       InstanceMethod("transpose", &ModelBuilder::Transpose, napi_enumerable),
       InstanceMethod("createModel", &ModelBuilder::CreateModel,
                      napi_enumerable)});
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("ModelBuilder", func);
  return exports;
}
