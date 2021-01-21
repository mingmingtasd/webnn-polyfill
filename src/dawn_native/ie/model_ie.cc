
#include "dawn_native/ie/model_ie.h"

#include <vector>

#include "common/Assert.h"
#include "common/Log.h"
#include "dawn_native/ErrorData.h"
#include "dawn_native/NamedOperands.h"
#include "dawn_native/ie/compilation_ie.h"
#include "error_ie.h"
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

Model::Model(ModelBuilder *model_builder) : ModelBase(model_builder) {
  // Load ienn_c_api.dll to compile the model.
  IEStatusCode code = IE(ie_create_model)(&ie_model_);
  if (code != IEStatusCode::OK) {
    dawn::ErrorLog() << "Failing to load ienn_c_api.dll.";
    return;
  }
}

Model::~Model() { IE(ie_model_free)(ie_model_); }

MaybeError Model::AddConstant(const op::Constant *constant) {
  ie_operand_descriptor ie_desc = ConvertTo(constant->GetOperandDescriptor());
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_constant)(ie_model_, &ie_desc, constant->GetValue(),
                                constant->GetSize(), &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add constant"));

  operand_id_map_[constant] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddInput(const op::Input *input) {
  ie_operand_descriptor ie_desc = ConvertTo(input->GetOperandDescriptor());
  ie_operand_t *ie_operand;
  IEStatusCode code = IE(ie_model_add_input)(ie_model_, &ie_desc, &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add input"));

  operand_id_map_[input] = std::string(ie_operand->name);
  input_id_map_[input->GetName()] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddOutput(const std::string &name,
                            const OperandBase *output) {
  ie_operand_t ie_operand;
  ie_operand.name = const_cast<char *>(operand_id_map_[output].c_str());
  IEStatusCode code = IE(ie_model_add_output)(ie_model_, &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add output"));

  output_name_map_[ie_operand.name] = name;
  return {};
}

MaybeError Model::AddBinary(const op::Binary *binary) {
  auto inputs = binary->Inputs();
  ie_operand_t primary;
  primary.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t secondary;
  secondary.name = const_cast<char *>(operand_id_map_[inputs[1].Get()].c_str());
  ie_operand_t *ie_operand = nullptr;
  IEStatusCode code = NOT_FOUND;
  if (binary->GetType() == op::BinaryOpType::kMatMul) {
    code =
        IE(ie_model_add_mat_mul)(ie_model_, &primary, &secondary, &ie_operand);
  } else {
    code = IE(ie_model_add_binary)(
        ie_model_, static_cast<ie_binary_type>(binary->GetType()), &primary,
        &secondary, &ie_operand);
  }
  DAWN_TRY(CheckStatusCode(code, "IE add binary"));

  operand_id_map_[binary] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddConv2d(const op::Conv2d *conv2d) {
  auto inputs = conv2d->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t filter;
  filter.name = const_cast<char *>(operand_id_map_[inputs[1].Get()].c_str());
  ie_operand_t *ie_operand;
  ie_conv2d_options_t ie_options = Conv2dOptionsForIE(conv2d->GetOptions());
  IEStatusCode code = IE(ie_model_add_conv2d)(ie_model_, &input, &filter,
                                              &ie_options, &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add conv2d"));

  operand_id_map_[conv2d] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddPool2d(const op::Pool2d *pool2d) {
  auto inputs = pool2d->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t *ie_operand;
  ie_pool2d_options_t ie_options = Pool2dOptionsForIE(pool2d->GetOptions());
  IEStatusCode code = IE(ie_model_add_pool2d)(
      ie_model_, static_cast<ie_pool_type>(pool2d->GetType()), &input,
      &ie_options, &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add pool2d"));

  operand_id_map_[pool2d] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddUnary(const op::Unary *unary) {
  auto inputs = unary->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t *ie_operand = nullptr;
  IEStatusCode code = NOT_FOUND;
  if (unary->GetType() == op::UnaryOpType::kRelu) {
    code = IE(ie_model_add_relu)(ie_model_, &input, &ie_operand);
  } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
    code = IE(ie_model_add_softmax)(ie_model_, &input, &ie_operand);
  }
  DAWN_TRY(CheckStatusCode(code, "IE add unary"));

  operand_id_map_[unary] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddReshape(const op::Reshape *reshape) {
  auto inputs = reshape->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t *ie_operand;
  IEStatusCode code =
      IE(ie_model_add_reshape)(ie_model_, &input, reshape->GetNewShape(),
                               reshape->GetNewShapeCount(), &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add reshape"));

  operand_id_map_[reshape] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::AddTranspose(const op::Transpose *transpose) {
  auto inputs = transpose->Inputs();
  ie_operand_t input;
  input.name = const_cast<char *>(operand_id_map_[inputs[0].Get()].c_str());
  ie_operand_t *ie_operand;
  ie_transpose_options_t ie_options =
      TransposeOptionsForIE(transpose->GetOptions());
  IEStatusCode code =
      IE(ie_model_add_transpose)(ie_model_, &input, &ie_options, &ie_operand);
  DAWN_TRY(CheckStatusCode(code, "IE add transpose"));

  operand_id_map_[transpose] = std::string(ie_operand->name);
  return {};
}

MaybeError Model::Finish() {
  IEStatusCode code = IE(ie_model_finish)(ie_model_);
  DAWN_TRY(CheckStatusCode(code, "IE finish creating model"));
  return {};
}

void Model::CompileImpl(WNNCompileCallback callback, void *userdata,
                        CompilationOptions const *options) {
  Compilation* compilation = new Compilation(this);
  compilation->Compile(callback, userdata, options);
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

std::string Model::GetOutputId(size_t index) {
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

} // namespace ie

} // namespace dawn_native
