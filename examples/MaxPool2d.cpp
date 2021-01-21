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

class MaxPool2d : public utils::WrappedModel {
public:
  MaxPool2d() {
    options_.windowDimensions = nullptr;
    options_.padding = nullptr;
    options_.strides = nullptr;
    options_.dilations = nullptr;
  }
  wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
    wnn::Operand input = nn.Input("input", InputDesc());
    return nn.MaxPool2d(input, &options_);
  }
  void SetPadding(std::vector<int32_t> padding) {
    padding_ = std::move(padding);
    options_.padding = padding_.data();
  }
  void SetWindowDimensions(std::vector<int32_t> dimensions) {
    window_dimensios_ = std::move(dimensions);
    options_.windowDimensions = window_dimensios_.data();
  }
private:
  wnn::Pool2dOptions options_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> window_dimensios_;
};

int main(int argc, const char* argv[]) {
  MaxPool2d *max_pool2d = new MaxPool2d();
  max_pool2d->SetInput({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  max_pool2d->SetWindowDimensions({3, 3});
  max_pool2d->SetExpectedBuffer({11, 12, 15, 16});
  max_pool2d->SetExpectedShape({1, 1, 2, 2});
  utils::Test(max_pool2d);
}
