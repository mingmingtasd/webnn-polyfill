// Copyright 2017 The Dawn Authors
//
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

#include "SampleUtils.h"

#include <vector>

class ReshapeModel : public utils::WrappedModel {
public:
  ReshapeModel() {}
  wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
    wnn::Operand input = nn.Input("input", InputDesc());
    return nn.Reshape(input, new_shape_.data(), new_shape_.size());
  }
  void SetNewShape(std::vector<int32_t> new_shape) {
    new_shape_ = std::move(new_shape);
  }
private:
  std::vector<int32_t> new_shape_;
};

int main(int argc, const char* argv[]) {
  ReshapeModel *reshape = new ReshapeModel();
  reshape->SetInput({2, 4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  reshape->SetNewShape({4, 2, 3});
  reshape->SetExpectedBuffer({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  reshape->SetExpectedShape({4, 2, 3});
  utils::Test(reshape);
}
