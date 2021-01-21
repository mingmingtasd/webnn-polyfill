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

#include "common/Log.h"

#include <vector>

class Transpose : public utils::WrappedModel {
public:
  Transpose() {
    options_.permutation = nullptr;
  }
  wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
    wnn::Operand input = nn.Input("input", InputDesc());
    if (!permutation_.empty()) {
      options_.permutation = permutation_.data();
      options_.permutationCount = permutation_.size();
    }
    return nn.Transpose(input, &options_);
  }
  void SetPermutation(std::vector<int32_t> permutation) {
    permutation_ = std::move(permutation);
  }
private:
  wnn::TransposeOptions options_;
  std::vector<int32_t> permutation_;
};

int main(int argc, const char* argv[]) {
  dawn::InfoLog() << "transpose default";
  Transpose *transpose_default = new Transpose();
  transpose_default->SetInput({2, 3, 4}, {0.43376675, 0.264609  , 0.26321858, 0.04260185, 0.6862414 ,0.26150206, 0.04169406, 0.24857993, 0.14914423, 0.19905873,0.33851373, 0.74131566, 0.91501445, 0.21852633, 0.02267954,0.22069663, 0.95799077, 0.17188412, 0.09732241, 0.03296741,0.04709655, 0.50648814, 0.13075736, 0.82511896});
  transpose_default->SetExpectedBuffer({0.43376675, 0.91501445, 0.6862414 , 0.95799077, 0.14914423,0.04709655, 0.264609  , 0.21852633, 0.26150206, 0.17188412,0.19905873, 0.50648814, 0.26321858, 0.02267954, 0.04169406,0.09732241, 0.33851373, 0.13075736, 0.04260185, 0.22069663,0.24857993, 0.03296741, 0.74131566, 0.82511896});
  transpose_default->SetExpectedShape({4, 3, 2});
  utils::Test(transpose_default);

  dawn::InfoLog() << "transpose permutations 1";
  Transpose *transpose_permutations_1 = new Transpose();
  transpose_permutations_1->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_1->SetPermutation({0, 1, 2});
  transpose_permutations_1->SetExpectedBuffer({
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_1->SetExpectedShape({2, 3, 4});
  utils::Test(transpose_permutations_1);

  dawn::InfoLog() << "transpose permutations 2";
  Transpose *transpose_permutations_2 = new Transpose();
  transpose_permutations_2->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_2->SetPermutation({0, 2, 1});
  transpose_permutations_2->SetExpectedBuffer({
    0.7760998,  0.8190919,  0.9306249,  0.8363521,  0.83241564, 0.00480607,
    0.10145967, 0.39479077, 0.39600816, 0.00533229, 0.5622921,  0.35415828,
    0.43689877, 0.4834097,  0.896649,   0.7603583,  0.6982117,  0.13060148,
    0.14368972, 0.7195266,  0.07824122, 0.11940759, 0.72893023, 0.33766487,
  });
  transpose_permutations_2->SetExpectedShape({2, 4, 3});
  utils::Test(transpose_permutations_2);

  dawn::InfoLog() << "transpose permutations 3";
  Transpose *transpose_permutations_3 = new Transpose();
  transpose_permutations_3->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_3->SetPermutation({1, 0, 2});
  transpose_permutations_3->SetExpectedBuffer({
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.43689877, 0.7603583,
    0.14368972, 0.11940759, 0.8190919,  0.83241564, 0.39479077, 0.5622921,
    0.4834097,  0.6982117,  0.7195266,  0.72893023, 0.9306249,  0.00480607,
    0.39600816, 0.35415828, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_3->SetExpectedShape({3, 2, 4});
  utils::Test(transpose_permutations_3);

  dawn::InfoLog() << "transpose permutations 4";
  Transpose *transpose_permutations_4 = new Transpose();
  transpose_permutations_4->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_4->SetPermutation({1, 2, 0});
  transpose_permutations_4->SetExpectedBuffer({
    0.7760998,  0.43689877, 0.8363521,  0.7603583,  0.10145967, 0.14368972,
    0.00533229, 0.11940759, 0.8190919,  0.4834097,  0.83241564, 0.6982117,
    0.39479077, 0.7195266,  0.5622921,  0.72893023, 0.9306249,  0.896649,
    0.00480607, 0.13060148, 0.39600816, 0.07824122, 0.35415828, 0.33766487,
  });
  transpose_permutations_4->SetExpectedShape({3, 4, 2});
  utils::Test(transpose_permutations_4);

  dawn::InfoLog() << "transpose permutations 5";
  Transpose *transpose_permutations_5 = new Transpose();
  transpose_permutations_5->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_5->SetPermutation({2, 0, 1});
  transpose_permutations_5->SetExpectedBuffer({
    0.7760998,  0.8190919,  0.9306249,  0.43689877, 0.4834097,  0.896649,
    0.8363521,  0.83241564, 0.00480607, 0.7603583,  0.6982117,  0.13060148,
    0.10145967, 0.39479077, 0.39600816, 0.14368972, 0.7195266,  0.07824122,
    0.00533229, 0.5622921,  0.35415828, 0.11940759, 0.72893023, 0.33766487,
  });
  transpose_permutations_5->SetExpectedShape({4, 2, 3});
  utils::Test(transpose_permutations_5);

  dawn::InfoLog() << "transpose permutations 6";
  Transpose *transpose_permutations_6 = new Transpose();
  transpose_permutations_6->SetInput({2, 3, 4}, {
    0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
    0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
    0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
    0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
  });
  transpose_permutations_6->SetPermutation({2, 1, 0});
  transpose_permutations_6->SetExpectedBuffer({
    0.7760998,  0.43689877, 0.8190919,  0.4834097,  0.9306249,  0.896649,
    0.8363521,  0.7603583,  0.83241564, 0.6982117,  0.00480607, 0.13060148,
    0.10145967, 0.14368972, 0.39479077, 0.7195266,  0.39600816, 0.07824122,
    0.00533229, 0.11940759, 0.5622921,  0.72893023, 0.35415828, 0.33766487,
  });
  transpose_permutations_6->SetExpectedShape({4, 3, 2});
  utils::Test(transpose_permutations_6);
}
