#include "ModelBuilder.h"

#include <iostream>
#include <vector>

#include "Model.h"
#include "NeuralNetworkContext.h"
#include "Operand.h"
#include "ops/Constant.h"
#include "ops/Conv2d.h"
#include "ops/Input.h"
#include "ops/Pool2d.h"
#include "ops/Reshape.h"
#include "ops/Transpose.h"

Napi::FunctionReference ModelBuilder::constructor;

WebnnNamedOperands GetNamedOperands(Napi::Object obj) {
  WebnnNamedOperands named_operands = webnn_native::CreateNamedOperands();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    Napi::Object item = obj.Get(name).As<Napi::Object>();
    if (!item.InstanceOf(Operand::constructor.Value())) {
      std::cout << "Expected 'Operand' for 'NamedOperand'" << std::endl;
      return named_operands;
    }
    WebnnOperand operand = Napi::ObjectWrap<Operand>::Unwrap(item)->GetOperand();
    webnnNamedOperandsSet(named_operands, name.data(), operand);
  }
  return named_operands;
}

ModelBuilder::ModelBuilder(const Napi::CallbackInfo& info) : 
    Napi::ObjectWrap<ModelBuilder>(info) {
  Napi::Object context = info[0].As<Napi::Object>();
  NeuralNetworkContext *unwrapped =
      Napi::ObjectWrap<NeuralNetworkContext>::Unwrap(context);
  model_builder_ =
      webnnNeuralNetworkContextCreateModelBuilder(unwrapped->GetContext());
}

ModelBuilder::~ModelBuilder() {
  webnnModelBuilderRelease(model_builder_);
}

Napi::Value ModelBuilder::Constant(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Constant>(info);
  node->SetOutput(webnnModelBuilderConstant(model_builder_,
                                          node->GetOperandDescriptor(),
                                          node->GetValue(), node->GetSize()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Input(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Input>(info);
  node->SetOutput(webnnModelBuilderInput(model_builder_, node->GetName().c_str(),
                                       node->GetOperandDescriptor()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Add(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(webnnModelBuilderAdd(model_builder_, node->GetInputs()[0],
                                     node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Mul(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(webnnModelBuilderMul(model_builder_, node->GetInputs()[0],
                                     node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::MatMul(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(webnnModelBuilderMatmul(model_builder_, node->GetInputs()[0],
                                        node->GetInputs()[1]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Conv2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Conv2d>(info);
  node->SetOutput(webnnModelBuilderConv2d(model_builder_, node->GetInputs()[0],
                                        node->GetInputs()[1],
                                        node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::MaxPool2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Pool2d>(info);
  node->SetOutput(webnnModelBuilderMaxPool2d(model_builder_, node->GetInputs()[0],
                                           node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::AveragePool2d(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Pool2d>(info);
  node->SetOutput(webnnModelBuilderAveragePool2d(
      model_builder_, node->GetInputs()[0], node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Relu(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(webnnModelBuilderRelu(model_builder_, node->GetInputs()[0]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Reshape(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Reshape>(info);
  node->SetOutput(webnnModelBuilderReshape(model_builder_, node->GetInputs()[0],
                                         node->GetNewShape().data(),
                                         node->GetNewShape().size()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Softmax(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Node>(info);
  node->SetOutput(webnnModelBuilderSoftmax(model_builder_, node->GetInputs()[0]));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::Transpose(const Napi::CallbackInfo &info) {
  Napi::Object object = Operand::constructor.New({});
  Operand *unwrapped = Napi::ObjectWrap<Operand>::Unwrap(object);
  auto node = std::make_shared<op::Transpose>(info);
  node->SetOutput(webnnModelBuilderTranspose(model_builder_, node->GetInputs()[0],
                                           node->GetOptions()));
  unwrapped->SetNode(node);
  return object;
}

Napi::Value ModelBuilder::CreateModel(const Napi::CallbackInfo &info) {
  WebnnNamedOperands named_operands =
      GetNamedOperands(info[0].As<Napi::Object>());
  std::vector<napi_value> args = {info[0].As<Napi::Value>()};
  Napi::Object model = Model::constructor.New(args);
  Model* unwrapped = Napi::ObjectWrap<Model>::Unwrap(model);
  unwrapped->SetModel(
      webnnModelBuilderCreateModel(model_builder_, named_operands));

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
