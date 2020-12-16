
#include "dawn_native/ModelBuilder.h"

#include <string>
#include <vector>
#include <stack>
#include <unordered_set>

#include "common/Assert.h"
#include "common/RefCounted.h"
#include "dawn_native/Operand.h"
#include "dawn_native/ops/binary.h"
#include "dawn_native/ops/constant.h"
#include "dawn_native/ops/conv2d.h"
#include "dawn_native/ops/input.h"
#include "dawn_native/ops/pool2d.h"
#include "dawn_native/ops/reshape.h"
#include "dawn_native/ops/transpose.h"
#include "dawn_native/ops/unary.h"

namespace dawn_native {

OperandBase *ModelBuilderBase::Constant(OperandDescriptor const *desc,
                                        void const *value, size_t size) {
  Ref<OperandBase> context = AcquireRef(new op::Constant(desc, value, size));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Input(char const *name,
                                     OperandDescriptor const *desc) {
  Ref<OperandBase> context = AcquireRef(new op::Input(std::string(name), desc));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Matmul(OperandBase *a, OperandBase *b) {
   Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kMatMul, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Add(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kAdd, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Mul(OperandBase *a, OperandBase *b) {
  Ref<OperandBase> context =
      AcquireRef(new op::Binary(op::BinaryOpType::kMul, a, b));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Conv2d(OperandBase *input, OperandBase *filter,
                                      Conv2dOptions const *options) {
  Ref<OperandBase> context = AcquireRef(new op::Conv2d(input, filter, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::AveragePool2d(OperandBase *input,
                                             Pool2dOptions const *options) {
  Ref<OperandBase> context = AcquireRef(
      new op::Pool2d(op::Pool2dType::kAveragePool2d, input, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::MaxPool2d(OperandBase *input,
                                         Pool2dOptions const *options) {
  Ref<OperandBase> context =
      AcquireRef(new op::Pool2d(op::Pool2dType::kMaxPool2d, input, options));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Relu(OperandBase *input) {
  Ref<OperandBase> context =
      AcquireRef(new op::Unary(op::UnaryOpType::kRelu, input));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Reshape(OperandBase *input,
                                       int32_t const *new_shape,
                                       size_t new_shape_count) {
  Ref<OperandBase> context =
      AcquireRef(new op::Reshape(input, new_shape, new_shape_count));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Softmax(OperandBase *input) {
  Ref<OperandBase> context =
      AcquireRef(new op::Unary(op::UnaryOpType::kSoftmax, input));
  return context.Detach();
}

OperandBase *ModelBuilderBase::Transpose(OperandBase *input,
                                         TransposeOptions const *options) {
  Ref<OperandBase> context = AcquireRef(new op::Transpose(input, options));
  return context.Detach();
}

ModelBase *ModelBuilderBase::CreateModel(NamedOperandsBase const *named_operands) {
  ModelBase* model = CreateModelImpl();
  std::vector<const OperandBase*> outputs;
  for (auto& named_output : named_operands->GetRecords()) {
    outputs.push_back(named_output.second);
  }
  std::vector<const OperandBase*> sorted_operands = TopologicalSort(outputs);
  for (auto& op : sorted_operands) {
    op->AddToModel(model);
  }
  for (auto& named_output : named_operands->GetRecords()) {
    model->AddOutput(named_output.first, named_output.second);
  }
  return model;
}

std::vector<const OperandBase*> ModelBuilderBase::TopologicalSort(
    std::vector<const OperandBase*>& root_nodes) {
  std::stack<const OperandBase*> nodes_to_do;
  std::unordered_set<const OperandBase*> nodes_done;
  std::vector<const OperandBase*> result;

  for (auto& node : root_nodes) {
    nodes_to_do.push(node);
  }
  while (nodes_to_do.size() > 0) {
    const OperandBase* node = nodes_to_do.top();
    if (nodes_done.count(node) == 0) {
      bool can_add = true;
      for (auto& dep : node->Inputs()) {
        if (nodes_done.count(dep.Get()) == 0){
          can_add = false;
          nodes_to_do.push(dep.Get());
        }
      }
      if (can_add) {
        result.push_back(node);
        nodes_to_do.pop();
        nodes_done.insert(node);
      }
    } else {
      nodes_to_do.pop();
    }
  }
  return result;
}

} // namespace dawn_native
