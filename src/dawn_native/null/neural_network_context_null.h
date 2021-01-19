#ifndef WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_
#define WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_

#include "dawn_native/NeuralNetworkContext.h"
#include "dawn_native/ModelBuilder.h"
#include "dawn_native/Model.h"
#include "dawn_native/Compilation.h"

namespace dawn_native {

namespace null {

// NeuralNetworkContext
class NeuralNetworkContext : public NeuralNetworkContextBase {
public:
  NeuralNetworkContext() = default;
  ~NeuralNetworkContext() override = default;

private:
  ModelBuilderBase *CreateModelBuilderImpl() override;
};

// ModelBuilder
class ModelBuilder : public ModelBuilderBase {
public:
  explicit ModelBuilder(NeuralNetworkContextBase *context);
  ~ModelBuilder() override = default;

private:
  ModelBase *CreateModelImpl() override;
};

// Model
class Model : public ModelBase {
public:
  explicit Model(ModelBuilder *model_builder);
  ~Model() override = default;
  virtual MaybeError AddConstant(const op::Constant *constant) override;
  virtual MaybeError AddInput(const op::Input *input) override;
  virtual MaybeError AddOutput(const std::string &name,
                               const OperandBase *ouput) override;
  virtual MaybeError AddBinary(const op::Binary *binary) override;
  virtual MaybeError AddConv2d(const op::Conv2d *conv2d) override;
  virtual MaybeError AddPool2d(const op::Pool2d *pool2d) override;
  virtual MaybeError AddReshape(const op::Reshape *relu) override;
  virtual MaybeError AddTranspose(const op::Transpose *transpose) override;
  virtual MaybeError AddUnary(const op::Unary *unary) override;
  virtual MaybeError Finish() override;

private:
  void CompileImpl(WNNCompileCallback callback, void *userdata,
                   CompilationOptions const *options) override;
};

// Compilation
class Compilation : public CompilationBase {
public:
  Compilation() = default;
  ~Compilation() override = default;

private:
  void ComputeImpl(NamedInputsBase *inputs, WNNComputeCallback callback,
                   void *userdata,
                   NamedOutputsBase *outputs = nullptr) override;
};

} // namespace null

} // namespace dawn_native

#endif // WEBNN_NATIVE_NULL_NEURAL_NETWORK_CONTEXT_NULL_H_
