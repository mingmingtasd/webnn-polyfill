// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef IE_NN_C_API_H
#define IE_NN_C_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
#define IENN_C_EXTERN extern "C"
#else
#define IENN_C_EXTERN
#endif

#if defined(__GNUC__) && (__GNUC__ < 4)
#define NEURAL_NETWORK_C_API(...) IENN_C_EXTERN __VA_ARGS__
#else
#if defined(_WIN32)
#ifdef IENN_c_wraper_EXPORTS
#define NEURAL_NETWORK_C_API(...) \
  IENN_C_EXTERN __declspec(dllexport) __VA_ARGS__ __cdecl
#else
#define NEURAL_NETWORK_C_API(...) \
  IENN_C_EXTERN __declspec(dllimport) __VA_ARGS__ __cdecl
#endif
#else
#define NEURAL_NETWORK_C_API(...) \
  IENN_C_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
#endif
#endif

/**
 * @enum IEStatusCode
 * @brief This enum contains codes for all possible return values of the
 * interface functions
 */
typedef enum {
  OK = 0,
  GENERAL_ERROR = -1,
  NOT_IMPLEMENTED = -2,
  NETWORK_NOT_LOADED = -3,
  PARAMETER_MISMATCH = -4,
  NOT_FOUND = -5,
  OUT_OF_BOUNDS = -6,
  /*
   * @brief exception not of std::exception derived type was thrown
   */
  UNEXPECTED = -7,
  REQUEST_BUSY = -8,
  RESULT_NOT_READY = -9,
  NOT_ALLOCATED = -10,
  INFER_NOT_STARTED = -11,
  NETWORK_NOT_READ = -12
} IEStatusCode;

typedef struct ie_operand {
  char* name;
} ie_operand_t;

typedef enum {
  PREFER_LOW_POWER = 0,
  PREFER_FAST_SINGLE_ANSWER = 1,
  PREFER_SUSTAINED_SPEED = 2,
  PREFER_ULTRA_LOW_POWER = 3,
} prefer_t;

enum ie_operand_type : uint32_t {
  Float32 = 0x00000000,
  Float16 = 0x00000001,
  Int32 = 0x00000002,
  Uint32 = 0x00000003,
};

typedef struct ie_operand_descriptor {
  ie_operand_type type;
  int32_t const* dimensions;
  uint32_t dimensionsCount = 0;
} ie_operand_descriptor_t;

enum ie_operand_layout : uint32_t {
  Nchw = 0x00000000,
  Nhwc = 0x00000001,
};

typedef struct ie_conv2d_options {
  uint32_t paddingCount = 4;
  int32_t const* padding;
  uint32_t stridesCount = 2;
  int32_t const* strides;
  uint32_t dilationsCount = 2;
  int32_t const* dilations;
  int32_t groups = 1;
  ie_operand_layout layout = ie_operand_layout::Nchw;
} ie_conv2d_options_t;

enum ie_pool_type {
  AVERAGE_POOL = 0,
  L2_POOL,
  MAX_POOL,
};

typedef struct ie_pool2d_options {
  uint32_t windowDimensionsCount = 2;
  int32_t const* windowDimensions;
  uint32_t paddingCount = 4;
  int32_t const* padding;
  uint32_t stridesCount = 2;
  int32_t const* strides;
  uint32_t dilationsCount = 2;
  int32_t const* dilations;
  ie_operand_layout layout = ie_operand_layout::Nchw;
} ie_pool2d_options_t;

typedef struct ie_transpose_options {
  uint32_t permutationCount = 0;
  int32_t const* permutation;
} ie_transpose_options_t;

typedef struct ie_model ie_model_t;
typedef struct ie_compilation ie_compilation_t;

enum ie_binary_type {
  ADD = 0,
  SUB,
  MUL,
  DIV,
  MAX,
  MIN,
};

/**
 * @struct dimensions
 * @brief Represents dimensions for data
 */
typedef struct dimensions {
  size_t ranks;
  int32_t* dims;
} ie_dimensions_t;

/**
 * @brief Create model. Use the ie_model_free() method to
 *  free the model memory.
 * @ingroup model
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode) ie_create_model(ie_model_t** model);

/**
 * @brief Releases memory occupied by model.
 * @ingroup model
 * @param operand A pointer to the model to free memory.
 */
NEURAL_NETWORK_C_API(void) ie_model_free(ie_model_t* model);

/**
 * @brief Add Constant node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_descriptor_t A pointer to the Operand Descriptor.
 * @param value The value of Operand.
 * @param size The size of the value.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_constant(ie_model_t* Compilation,
                      ie_operand_descriptor_t const* desc,
                      void const* value,
                      size_t length,
                      ie_operand_t**);

/**
 * @brief Add Input node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param name The name of Input Operand to Set Value with JS.
 * @param ie_operand_descriptor_t A pointer to the Operand Descriptor.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_input(ie_model_t* model,
                   ie_operand_descriptor_t const* desc,
                   ie_operand_t** operand);

/**
 * @brief Add Output with node name. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param operand Get the name of node to create output node.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_output(ie_model_t* model, ie_operand_t* operand);

/**
 * @brief Add MatMul node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The prmiary operand.
 * @param ie_operand_t The secondary operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_mat_mul(ie_model_t* model,
                     ie_operand_t* a,
                     ie_operand_t* b,
                     ie_operand_t** operand);

/**
 * @brief Add binary node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The prmiary operand.
 * @param ie_operand_t The secondary operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_binary(ie_model_t* model,
                    ie_binary_type type,
                    ie_operand_t* a,
                    ie_operand_t* b,
                    ie_operand_t** operand);

/**
 * @brief Add conv2d node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @param ie_operand_t The filter operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_conv2d(ie_model_t* model,
                    ie_operand_t* input,
                    ie_operand_t* filter,
                    ie_conv2d_options_t* options,
                    ie_operand_t** operand);

/**
 * @brief Add pool2d node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @param ie_operand_t The filter operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_pool2d(ie_model_t* model,
                    ie_pool_type type,
                    ie_operand_t* input,
                    ie_pool2d_options_t* options,
                    ie_operand_t** operand);

/**
 * @brief Add Relu node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_relu(ie_model_t* model,
                  ie_operand_t* input,
                  ie_operand_t** operand);

/**
 * @brief Add Reshape node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_reshape(ie_model_t* model,
                     ie_operand_t* input,
                     int32_t const* new_shape,
                     uint32_t new_shape_count,
                     ie_operand_t** operand);

/**
 * @brief Add Softmax node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_softmax(ie_model_t* model,
                     ie_operand_t* input,
                     ie_operand_t** operand);

/**
 * @brief Add transpose node to nGraph. Use the ie_operand_free() method to
 *  free the operand memory.
 * @ingroup model
 * @param ie_operand_t The input operand.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_add_transpose(ie_model_t* model,
                       ie_operand_t* input,
                       ie_transpose_options* options,
                       ie_operand_t** operand);

/**
 * @brief Releases memory occupied by operand.
 * @ingroup Operand
 * @param operand A pointer to the operand to free memory.
 */
NEURAL_NETWORK_C_API(void) ie_operand_free(ie_operand_t* operand);

/**
 * @brief Start to load the network to plugin.
 * @ingroup model
 * @param compliation A pointer to the specified ie_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode) ie_model_finish(ie_model_t* compliation);

/**
 * @brief Get outputs number.
 * @ingroup model
 * @param compliation A pointer to the specified ie_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_get_outputs_number(const ie_model_t* model, size_t* size_result);

/**
 * @brief Get output name with index.
 * @ingroup model
 * @param compliation A pointer to the specified ie_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_model_get_output_name(const ie_model_t* model,
                         const size_t number,
                         char** name);

/**
 * @brief Get output name with index.
 * @ingroup model
 * @param compliation A pointer to the specified ie_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode) ie_model_free_name(char** name);

/**
 * @brief Create compilation. Use the ie_compilation_free() method to
 *  free the compilation memory.
 * @ingroup compilation
 * @param ie_model_t The model need to be compiled.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_create_compilation(ie_model_t* model, ie_compilation_t** compilation);

/**
 * @brief Releases memory occupied by compilation.
 * @ingroup compilation
 * @param operand A pointer to the operand to free memory.
 */
NEURAL_NETWORK_C_API(void) ie_compilation_free(ie_compilation_t* compilation);

/**
 * @brief Set input data to compute.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_execution_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_set_input(ie_compilation_t* compilation,
                         ie_operand_t* operand,
                         const void* buffer,
                         uint32_t length);

/**
 * @brief Set output data for the model.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_execution_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_get_output(ie_compilation_t* compilation,
                          ie_operand_t* operand,
                          void* buffer,
                          uint32_t length);

/**
 * @brief Compute the compiled mode.
 * @ingroup compilation
 * @param ie_compilation_t the compilation.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_compute(ie_compilation_t* compilation);

/**
 * @brief Get buffer with name.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_get_buffer(const ie_compilation_t* compilation,
                          const char* name,
                          void** buffer,
                          size_t* byte_length);

/**
 * @brief free the buffer.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode) ie_compilation_free_buffer(void** buffer);

/**
 * @brief Get output dimensions with name.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_get_dimensions(const ie_compilation_t* compilation,
                              const char* name,
                              ie_dimensions_t* dimensions);

/**
 * @brief free the output dimensions.
 * @ingroup compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
NEURAL_NETWORK_C_API(IEStatusCode)
ie_compilation_free_dimensions(ie_dimensions_t* dimensions);

#endif  // IE_NN_C_API_H
