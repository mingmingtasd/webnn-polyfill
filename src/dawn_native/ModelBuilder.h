#ifndef WEBNN_NATIVE_MODEL_BUILDER_H_
#define WEBNN_NATIVE_MODEL_BUILDER_H_

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class ModelBuilderBase : public RefCounted {
public:
  ModelBuilderBase() = default;
  virtual ~ModelBuilderBase() = default;

  // DAWN API
  OperandBase *Constant(OperandDescriptor const *desc, void const *value,
                        size_t size);
  OperandBase *Input(char const *name, OperandDescriptor const *desc);
  OperandBase *Matmul(OperandBase *a, OperandBase *b);
  OperandBase *Add(OperandBase *, OperandBase *);
  OperandBase *Mul(OperandBase *, OperandBase *);
  OperandBase *Conv2d(OperandBase *, OperandBase *,
                      Conv2dOptions const *options);
  OperandBase *AveragePool2d(OperandBase *, Pool2dOptions const *options);
  OperandBase *MaxPool2d(OperandBase *, Pool2dOptions const *options);
  OperandBase *Relu(OperandBase *);
  OperandBase *Reshape(OperandBase *, int32_t const *, size_t);
  OperandBase *Softmax(OperandBase *);
  ModelBase *CreateModel(NamedOperand const *named_operand, size_t size);

private:
  virtual ModelBase *CreateModelImpl(NamedOperand const *named_operand,
                                     size_t size) = 0;
};

} // namespace dawn_native

#endif // WEBNN_NATIVE_MODEL_BUILDER_H_
