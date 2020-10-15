
#include "dawn_native/ie/model_ie.h"

#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/ie/compilation_ie.h"
#include "ienn_symbol_table.h"

namespace dawn_native {

namespace ie {

namespace {
ie_operand_descriptor ConvertTo(OperandDescriptor const *desc) {
  ie_operand_descriptor ie_desc;
  ie_desc.dimensions = desc->dimensions;
  ie_desc.dimensionsCount = desc->dimensionsCount;
  switch (desc->type) {
  case wnn::OperandType::Float32:
    ie_desc.type = ie_operand_type::Float32;
    break;
  case wnn::OperandType::Int32:
    ie_desc.type = ie_operand_type::Int32;
    break;
  case wnn::OperandType::Float16:
    ie_desc.type = ie_operand_type::Float16;
    break;
  case wnn::OperandType::Uint32:
    ie_desc.type = ie_operand_type::Uint32;
    break;
  default:
    UNREACHABLE();
  }
  return ie_desc;
}
} // namespace

Model::Model(struct NamedOperand const *named_operands, size_t size) {
  // Load ienn_c_api.dll to compile the model.
  IEStatusCode code = IE(ie_create_model)(&ie_model_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to load ienn_c_api.dll.";
    return;
  }
  for (uint32_t i = 0; i < size; ++i) {
    struct NamedOperand output = named_operands[i];
    BuildNeuralNetworkModel(output.operand);
    // Add output node to ngraph.
    AddOutput(output.operand);
    // Insert the named operand to map.
    named_operands_[std::string(output.name)] = output.operand;
  }
  // Finish to create the model that is CNNNetwork.
  Finish();
}

// Traversal graph inorder to create model.
void Model::BuildNeuralNetworkModel(OperandBase *root) {
  // the stack is used for traversaling model tree.
  std::vector<Ref<OperandBase>> stack;
  OperandBase *operand = root;
  while (operand || !stack.empty()) {
    while (operand) {
      stack.push_back(operand);
      operand = operand->FirstInput().Get();
    }
    // It will be input/constant if there is no FirstInput operand.
    if (!stack.empty()) {
      // The index_ in Operand will be add when calling NextInput, so keep
      // the line.
      OperandBase *next_input = stack.back()->NextInput().Get();
      if (next_input && !next_input->Traversal()) {
        operand = next_input;
      } else {
        // Call the AddLayer virtual function to Add the operand with Android NN
        // API.
        stack.back()->AddOperand(this);
        // Set the operand as traversalled so that it doesn't push again.
        stack.back()->SetTraversal(true);
        // Pop the node.
        stack.pop_back();
        // Don't use the code [operand = stack.back()] so that Origin branch
        // [while (operand)] will be not traversalled again.
        // operand = stack.back();
      }
    }
  }
}

void Model::AddConstant(OperandBase *constant, OperandDescriptor const *desc,
                        void const *value, size_t size) {
  ie_operand_descriptor ie_desc = ConvertTo(desc);
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_constant)(ie_model_, &ie_desc, value, size, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add constant, the code is " << code << ".";
    return;
  }
  constant->SetName(std::string(ie_operand->name));
}

void Model::AddInput(OperandBase *input, const std::string name,
                     OperandDescriptor const *desc) {
  ie_operand_descriptor ie_desc = ConvertTo(desc);
  ie_operand_t *ie_operand;
  IEStatusCode code = IE(ie_model_add_input)(ie_model_, &ie_desc, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add input , the code is " << code << ".";
    return;
  }
  input->SetName(std::string(ie_operand->name));
  named_operands_[name] = input;
}

void Model::AddOutput(OperandBase *ouput) {
  ie_operand_t ie_operand;
  ie_operand.name = const_cast<char *>(ouput->GetName().c_str());
  IEStatusCode code = IE(ie_model_add_output)(ie_model_, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add input , the code is " << code << ".";
    return;
  }
}

void Model::Finish() {
  IEStatusCode code = IE(ie_model_finish)(ie_model_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to finish the model.";
    return;
  }
}

void Model::AddMatMul(OperandBase *mutmul, OperandBase *a, OperandBase *b) {
  ie_operand_t primary;
  primary.name = const_cast<char *>(a->GetName().c_str());
  ie_operand_t secondary;
  secondary.name = const_cast<char *>(b->GetName().c_str());
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_mat_mul)(ie_model_, &primary, &secondary, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add matmul, the code is " << code << ".";
    return;
  }
  mutmul->SetName(std::string(ie_operand->name));
}

OperandBase *Model::GetNamedOperand(std::string name) {
  return named_operands_[name];
}

void Model::CompileImpl(WNNCompileCallback callback,
                        CompilationOptions const *options) {
  callback(reinterpret_cast<WNNCompilation>(new Compilation(this)));
}

ie_model_t *Model::GetInferenceEngineModel() { return ie_model_; }

} // namespace ie

} // namespace dawn_native
