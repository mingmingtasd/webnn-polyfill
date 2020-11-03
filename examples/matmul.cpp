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

class MatMulModel : public utils::WrappedModel {
public:
  MatMulModel() {}
  wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
    wnn::Operand input = nn.Input("input", InputDesc());
    wnn::Operand constant =
        nn.Constant(ConstantDesc(), ConstantBuffer(), product(ConstantShape()));
    return nn.Matmul(input, constant);
  }
};

int main(int argc, const char* argv[]) {
  MatMulModel *mat_mul = new MatMulModel();
  mat_mul->SetInput({3, 4}, {0.9602246, 0.97682184, -0.33201018, 0.8248904, 0.40872088, 0.18995902, 0.69355214, -0.37210146, 0.18104352, 3.270753, -0.803097, -0.7268995});
  mat_mul->SetConstant({4, 3}, {0.17467105, -1.2045133, -0.02621938, 0.6096196, 1.4499376, 1.3465316, 0.03289436, 1.0754977, -0.61485314, 0.94857556, -0.36462623, 1.402278});
  mat_mul->SetExpectedBuffer({1.5347629, -0.3981255, 2.6510081, -0.14295794, 0.6647107, -0.70315295, 1.3096018, 3.9256358, 3.873897});
  utils::Test(mat_mul);
}
