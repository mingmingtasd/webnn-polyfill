// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "webnn_native/ops/Pool2d.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    Pool2d::Pool2d(ModelBuilderBase* builder,
                   Pool2dType type,
                   OperandBase* input,
                   Pool2dOptions const* options)
        : OperandBase(builder, {input}), op_type_(type) {
        if (options == nullptr || options->windowDimensions == nullptr) {
            window_dimensions_ = std::vector<int32_t>(2, 1);
        } else {
            window_dimensions_.assign(options->windowDimensions,
                                      options->windowDimensions + options->windowDimensionsCount);
        }
        options_.windowDimensions = window_dimensions_.data();
        options_.windowDimensionsCount = window_dimensions_.size();

        if (options == nullptr || options->padding == nullptr) {
            padding_ = std::vector<int32_t>(4, 0);
        } else {
            padding_.assign(options->padding, options->padding + options->paddingCount);
        }
        options_.padding = padding_.data();
        options_.paddingCount = padding_.size();

        if (options == nullptr || options->strides == nullptr) {
            stride_ = std::vector<int32_t>(2, 1);
        } else {
            stride_.assign(options->strides, options->strides + options->stridesCount);
        }
        options_.strides = stride_.data();
        options_.stridesCount = stride_.size();

        if (options == nullptr || options->dilations == nullptr) {
            dilations_ = std::vector<int32_t>(2, 1);
        } else {
            dilations_.assign(options->dilations, options->dilations + options->dilationsCount);
        }
        options_.dilations = dilations_.data();
        options_.dilationsCount = dilations_.size();

        if (options == nullptr) {
            options_.layout = webnn::OperandLayout::Nchw;
        } else {
            options_.layout = options->layout;
        }
    }

    MaybeError Pool2d::AddToModel(ModelBase* model) const {
        return model->AddPool2d(this);
    }

    Pool2dOptions const* Pool2d::GetOptions() const {
        return &options_;
    }

    Pool2dType Pool2d::GetType() const {
        return op_type_;
    }

    MaybeError Pool2d::ValidateAndInferTypes() {
        auto input = inputs_[0];
        if (input->IsError()) {
            return DAWN_VALIDATION_ERROR("Argument input is invalid.");
        }
        // The input 4-D tensor
        if (input->Rank() != 4) {
            return DAWN_VALIDATION_ERROR("Argument input is not a 4D tensor.");
        }
        // windowDimensions: a sequence of long of length 2
        if (options_.windowDimensionsCount != 2) {
            return DAWN_VALIDATION_ERROR("windowDimensionsCount is incorrect.");
        }
        // padding: a sequence of long of length 4
        if (options_.paddingCount != 4) {
            return DAWN_VALIDATION_ERROR("paddingCount is incorrect.");
        }
        // strides: a sequence of long of length 2
        if (options_.stridesCount != 2) {
            return DAWN_VALIDATION_ERROR("stridesCount is incorrect.");
        }
        // dilations: a sequence of long of length 2.
        if (options_.dilationsCount != 2) {
            return DAWN_VALIDATION_ERROR("dilationsCount is incorrect.");
        }

        type_ = input->Type();
        rank_ = 4;

        return {};
    }

}}  // namespace webnn_native::op