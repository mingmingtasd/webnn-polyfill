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

class Transpose : public utils::WrappedModel {
public:
  Transpose() {
    options_.permutation = nullptr;
  }
  wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
    wnn::Operand input = nn.Input("input", InputDesc());
    return nn.Transpose(input, &options_);
  }
private:
  wnn::TransposeOptions options_;
};

int main(int argc, const char* argv[]) {
  Transpose *conv2d = new Transpose();
  conv2d->SetInput({2, 3, 4}, {0.43376675, 0.264609  , 0.26321858, 0.04260185, 0.6862414 ,0.26150206, 0.04169406, 0.24857993, 0.14914423, 0.19905873,0.33851373, 0.74131566, 0.91501445, 0.21852633, 0.02267954,0.22069663, 0.95799077, 0.17188412, 0.09732241, 0.03296741,0.04709655, 0.50648814, 0.13075736, 0.82511896});
  conv2d->SetExpectedBuffer({0.43376675, 0.91501445, 0.6862414 , 0.95799077, 0.14914423,0.04709655, 0.264609  , 0.21852633, 0.26150206, 0.17188412,0.19905873, 0.50648814, 0.26321858, 0.02267954, 0.04169406,0.09732241, 0.33851373, 0.13075736, 0.04260185, 0.22069663,0.24857993, 0.03296741, 0.74131566, 0.82511896});
  utils::Test(conv2d);
}
