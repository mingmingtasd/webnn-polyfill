
#include "dawn_native/ie/model_ie.h"

#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/ie/compilation_ie.h"
#include "dawn_native/NamedOperands.h"
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

ie_conv2d_options Conv2dOptionsForIE(Conv2dOptions const *options) {
  ie_conv2d_options ie_options;
  ie_options.padding = options->padding;
  ie_options.strides = options->strides;
  ie_options.dilations = options->dilations;
  ie_options.groups = options->groups;
  ie_options.layout = static_cast<ie_operand_layout>(options->layout);
  return ie_options;
}

ie_transpose_options TransposeOptionsForIE(TransposeOptions const *options) {
  if (options == nullptr)
    return {};
  ie_transpose_options ie_options;
  ie_options.permutation = options->permutation;
  ie_options.permutationCount = options->permutationCount;
  return ie_options;
}

ie_pool2d_options Pool2dOptionsForIE(Pool2dOptions const *options) {
  ie_pool2d_options ie_options;
  ie_options.windowDimensions = options->windowDimensions;
  ie_options.padding = options->padding;
  ie_options.strides = options->strides;
  ie_options.dilations = options->dilations;
  ie_options.layout = static_cast<ie_operand_layout>(options->layout);
  return ie_options;
}

} // namespace

Model::Model(NamedOperandsBase const *named_operands) {
  // Load ienn_c_api.dll to compile the model.
  IEStatusCode code = IE(ie_create_model)(&ie_model_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to load ienn_c_api.dll.";
    return;
  }

  
  for (auto &itr : named_operands->GetRecords()) {
    named_operands_[itr.first] = itr.second;
    BuildNeuralNetworkModel(itr.second);
    // Add output node to ngraph.
    AddOutput(itr.second);
    user_name_map_[itr.second->GetName()] = itr.first;
  }

  
  // Finish to create the model that is CNNNetwork.
  Finish();
}

// Traversal graph inorder to create model.
void Model::BuildNeuralNetworkModel(const OperandBase *root) {
  if (!root)
    return;
  // the stack is used for traversaling model tree.
  std::vector<Ref<OperandBase>> stack;
  // Push back the root node so that only child node need to be add to stack in
  // secondary while loop.
  stack.push_back(const_cast<OperandBase*>(root));
  traversalled_.insert(root);
  const OperandBase *operand = root;
  while (!stack.empty()) {
    while (operand) {
      bool sub_graph = false;
      for (auto &input : operand->Inputs()) {
        // Push back the operand if it's not traversalled.
        if (traversalled_.find(input.Get()) == traversalled_.end()) {
          stack.push_back(input);
          traversalled_.insert(input.Get());
          // traversal next sub graph with the operand that isn't nullptr;
          if (!sub_graph) {
            operand = input.Get();
            sub_graph = true;
          }
        }
      }
      // the sub graph of the operand has been add to model or input/constant.
      if (!sub_graph) {
        operand = nullptr;
      }
    }
    // The sub graph of the operand has been add to model by native api or the
    // the operand is input/constant, then add current operand with AddOperand
    // virtual function.
    stack.back()->AddToModel(this);
    // Pop the node.
    stack.pop_back();

    // Continue to traversal unused operand.
    operand = stack.back().Get();
  }
}

void Model::AddConstant(op::Constant *constant) {
  ie_operand_descriptor ie_desc = ConvertTo(constant->GetOperandDescriptor());
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_constant)(ie_model_, &ie_desc, constant->GetValue(),
                                constant->GetSize(), &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add constant, the code is " << code << ".";
    return;
  }
  constant->SetName(std::string(ie_operand->name));
}

void Model::AddInput(op::Input *input) {
  ie_operand_descriptor ie_desc = ConvertTo(input->GetOperandDescriptor());
  ie_operand_t *ie_operand;
  IEStatusCode code = IE(ie_model_add_input)(ie_model_, &ie_desc, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add input , the code is " << code << ".";
    return;
  }
  input->SetName(std::string(ie_operand->name));
  named_operands_[input->GetName()] = input;
}

void Model::AddOutput(const OperandBase *ouput) {
  ie_operand_t ie_operand;
  ie_operand.name = const_cast<char *>(ouput->GetName().c_str());
  IEStatusCode code = IE(ie_model_add_output)(ie_model_, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add input , the code is " << code << ".";
    return;
  }
}

void Model::AddBinary(op::Binary *binary) {
  auto inputs = binary->Inputs();
  ie_operand_t primary;
  primary.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t secondary;
  secondary.name = const_cast<char *>(inputs[1]->GetName().c_str());
  ie_operand_t *ie_operand = nullptr;
  IEStatusCode code = NOT_FOUND;
  if (binary->GetType() == op::BinaryOpType::kMatMul) {
    code = IE(ie_model_add_mat_mul)(
        ie_model_, &primary, &secondary, &ie_operand);
  } else {
    code = IE(ie_model_add_binary)(
      ie_model_, static_cast<ie_binary_type>(binary->GetType()), &primary,
      &secondary, &ie_operand);
  }
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add binary, the code is " << code << ".";
    return;
  }
  binary->SetName(std::string(ie_operand->name));
}

void Model::AddConv2d(op::Conv2d *conv2d) {
  auto inputs = conv2d->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t filter;
  filter.name = const_cast<char *>(inputs[1]->GetName().c_str());
  ie_operand_t *ie_operand;
  ie_conv2d_options_t ie_options = Conv2dOptionsForIE(conv2d->Options());
  IEStatusCode code = IE(ie_model_add_conv2d)(ie_model_, &input, &filter,
                                              &ie_options, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add matmul, the code is " << code << ".";
    return;
  }
  conv2d->SetName(std::string(ie_operand->name));
}

void Model::AddPool2d(op::Pool2d *pool2d) {
  auto inputs = pool2d->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t *ie_operand;
  ie_pool2d_options_t ie_options = Pool2dOptionsForIE(pool2d->Options());
  IEStatusCode code = IE(ie_model_add_pool2d)(
      ie_model_, static_cast<ie_pool_type>(pool2d->GetType()), &input,
      &ie_options, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add matmul, the code is " << code << ".";
    return;
  }
  pool2d->SetName(std::string(ie_operand->name));
}

void Model::AddUnary(op::Unary *unary) {
  auto inputs = unary->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t *ie_operand = nullptr;
  IEStatusCode code = NOT_FOUND;
  if (unary->GetType() == op::UnaryOpType::kRelu) {
    code = IE(ie_model_add_relu)(ie_model_, &input, &ie_operand);
  } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
    code = IE(ie_model_add_softmax)(ie_model_, &input, &ie_operand);
  }
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add relu, the code is " << code << ".";
    return;
  }
  unary->SetName(std::string(ie_operand->name));
}

void Model::AddReshape(op::Reshape *reshape) {
  auto inputs = reshape->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_reshape)(ie_model_, &input, reshape->GetNewShape(),
                               reshape->GetNewShapeCount(), &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add relu, the code is " << code << ".";
    return;
  }
  reshape->SetName(std::string(ie_operand->name));
}

void Model::AddTranspose(op::Transpose *transpose) {
  auto inputs = transpose->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(inputs[0]->GetName().c_str());
  ie_operand_t *ie_operand;
  ie_transpose_options_t ie_options =
      TransposeOptionsForIE(transpose->Options());
  IEStatusCode code =
      IE(ie_model_add_transpose)(ie_model_, &input, &ie_options, &ie_operand);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to add transpose, the code is " << code << ".";
    return;
  }
  transpose->SetName(std::string(ie_operand->name));
}

void Model::Finish() {
  IEStatusCode code = IE(ie_model_finish)(ie_model_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to finish the model.";
    return;
  }
}

const OperandBase *Model::GetNamedOperand(std::string name) {
  return named_operands_[name];
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  callback(reinterpret_cast<WNNCompilation>(new Compilation(this)), userdata);
}

ie_model_t *Model::GetInferenceEngineModel() { return ie_model_; }

size_t Model::GetOutputsNumber() {
  size_t output_number = 0;
  IEStatusCode code =
      IE(ie_model_get_outputs_number)(ie_model_, &output_number);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to get output number for IE.";
  }
  return output_number;
}

std::string Model::GetOutputName(size_t index) {
  char *output_name;
  IEStatusCode code =
      IE(ie_model_get_output_name)(ie_model_, index, &output_name);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to get output name for IE.";
    return std::string();
  }
  std::string name(output_name);
  // The name has been kept in outputs object, so it can be free.
  IE(ie_model_free_name)(&output_name);

  return name;
}

const std::string& Model::GetUserName(const std::string& name) {
  return user_name_map_.at(name);
}

} // namespace ie

} // namespace dawn_native
