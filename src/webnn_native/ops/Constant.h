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

#ifndef WEBNN_NATIVE_OPS_CONSTANT_H_
#define WEBNN_NATIVE_OPS_CONSTANT_H_

#include "webnn_native/Model.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Constant final : public OperandBase {
      public:
        Constant(ModelBuilderBase* builder,
                 const OperandDescriptor* desc,
                 void const* value,
                 size_t size)
            : OperandBase(builder), value_(value), size_(size) {
            descriptor_.type = desc->type;
            dimensions_.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
            descriptor_.dimensions = dimensions_.data();
            descriptor_.dimensionsCount = dimensions_.size();
        }
        ~Constant() override = default;

        MaybeError AddToModel(ModelBase* model) const override {
            return model->AddConstant(this);
        }

        MaybeError Validate() override;
        const OperandDescriptor* GetOperandDescriptor() const {
            return &descriptor_;
        }
        void const* GetValue() const {
            return value_;
        }
        size_t GetSize() const {
            return size_;
        }

      private:
        OperandDescriptor descriptor_;
        std::vector<int32_t> dimensions_;
        void const* value_;
        size_t size_;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_CONSTANT_H_
