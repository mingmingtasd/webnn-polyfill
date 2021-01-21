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

#include "dawn_native/ops/Conv2d.h"

#include "common/Log.h"
#include "dawn_native/Error.h"

namespace dawn_native { namespace op {

    Conv2d::Conv2d(ModelBuilderBase* builder,
                   OperandBase* input,
                   OperandBase* filter,
                   Conv2dOptions const* options)
        : OperandBase(builder, {input, filter}) {
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

        options_.groups = options->groups;
        options_.layout = options->layout;
    }

    MaybeError Conv2d::AddToModel(ModelBase* model) const {
        return model->AddConv2d(this);
    }

    Conv2dOptions const* Conv2d::GetOptions() const {
        return &options_;
    }

    MaybeError Conv2d::Validate() {
        return {};
    }

}}  // namespace dawn_native::op
