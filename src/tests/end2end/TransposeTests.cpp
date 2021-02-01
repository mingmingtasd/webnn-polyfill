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

#include "examples/SampleUtils.h"
#include "gtest/gtest.h"

class TransposeTests : public testing::Test {};

class Transpose : public utils::WrappedModel {
  public:
    Transpose() : mOptions({}) {
    }
    wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
        wnn::Operand input = nn.Input("input", InputDesc());
        if (!mPermutation.empty()) {
            mOptions.permutation = mPermutation.data();
            mOptions.permutationCount = mPermutation.size();
        }
        return nn.Transpose(input, &mOptions);
    }
    void SetPermutation(std::vector<int32_t> permutation) {
        mPermutation = std::move(permutation);
    }

  private:
    wnn::TransposeOptions mOptions;
    std::vector<int32_t> mPermutation;
};

TEST_F(TransposeTests, TransposeDefault) {
    Transpose* transposeDefault = new Transpose();
    transposeDefault->SetInput(
        {2, 3, 4}, {0.43376675, 0.264609,   0.26321858, 0.04260185, 0.6862414,  0.26150206,
                    0.04169406, 0.24857993, 0.14914423, 0.19905873, 0.33851373, 0.74131566,
                    0.91501445, 0.21852633, 0.02267954, 0.22069663, 0.95799077, 0.17188412,
                    0.09732241, 0.03296741, 0.04709655, 0.50648814, 0.13075736, 0.82511896});
    transposeDefault->SetExpectedBuffer({0.43376675, 0.91501445, 0.6862414,  0.95799077, 0.14914423,
                                         0.04709655, 0.264609,   0.21852633, 0.26150206, 0.17188412,
                                         0.19905873, 0.50648814, 0.26321858, 0.02267954, 0.04169406,
                                         0.09732241, 0.33851373, 0.13075736, 0.04260185, 0.22069663,
                                         0.24857993, 0.03296741, 0.74131566, 0.82511896});
    transposeDefault->SetExpectedShape({4, 3, 2});
    EXPECT_TRUE(utils::Test(transposeDefault));
}

TEST_F(TransposeTests, TransposePermutations1) {
    Transpose* transposePermutations1 = new Transpose();
    transposePermutations1->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations1->SetPermutation({0, 1, 2});
    transposePermutations1->SetExpectedBuffer({
        0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
        0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
        0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
        0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
    });
    transposePermutations1->SetExpectedShape({2, 3, 4});
    EXPECT_TRUE(utils::Test(transposePermutations1));
}

TEST_F(TransposeTests, TransposePermutations2) {
    Transpose* transposePermutations2 = new Transpose();
    transposePermutations2->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations2->SetPermutation({0, 2, 1});
    transposePermutations2->SetExpectedBuffer({
        0.7760998,  0.8190919,  0.9306249,  0.8363521,  0.83241564, 0.00480607,
        0.10145967, 0.39479077, 0.39600816, 0.00533229, 0.5622921,  0.35415828,
        0.43689877, 0.4834097,  0.896649,   0.7603583,  0.6982117,  0.13060148,
        0.14368972, 0.7195266,  0.07824122, 0.11940759, 0.72893023, 0.33766487,
    });
    transposePermutations2->SetExpectedShape({2, 4, 3});
    EXPECT_TRUE(utils::Test(transposePermutations2));
}

TEST_F(TransposeTests, TransposePermutations3) {
    Transpose* transposePermutations3 = new Transpose();
    transposePermutations3->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations3->SetPermutation({1, 0, 2});
    transposePermutations3->SetExpectedBuffer({
        0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.43689877, 0.7603583,
        0.14368972, 0.11940759, 0.8190919,  0.83241564, 0.39479077, 0.5622921,
        0.4834097,  0.6982117,  0.7195266,  0.72893023, 0.9306249,  0.00480607,
        0.39600816, 0.35415828, 0.896649,   0.13060148, 0.07824122, 0.33766487,
    });
    transposePermutations3->SetExpectedShape({3, 2, 4});
    EXPECT_TRUE(utils::Test(transposePermutations3));
}

TEST_F(TransposeTests, TransposePermutations4) {
    Transpose* transposePermutations4 = new Transpose();
    transposePermutations4->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations4->SetPermutation({1, 2, 0});
    transposePermutations4->SetExpectedBuffer({
        0.7760998,  0.43689877, 0.8363521,  0.7603583,  0.10145967, 0.14368972,
        0.00533229, 0.11940759, 0.8190919,  0.4834097,  0.83241564, 0.6982117,
        0.39479077, 0.7195266,  0.5622921,  0.72893023, 0.9306249,  0.896649,
        0.00480607, 0.13060148, 0.39600816, 0.07824122, 0.35415828, 0.33766487,
    });
    transposePermutations4->SetExpectedShape({3, 4, 2});
    EXPECT_TRUE(utils::Test(transposePermutations4));
}

TEST_F(TransposeTests, TransposePermutations5) {
    Transpose* transposePermutations5 = new Transpose();
    transposePermutations5->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations5->SetPermutation({2, 0, 1});
    transposePermutations5->SetExpectedBuffer({
        0.7760998,  0.8190919,  0.9306249,  0.43689877, 0.4834097,  0.896649,
        0.8363521,  0.83241564, 0.00480607, 0.7603583,  0.6982117,  0.13060148,
        0.10145967, 0.39479077, 0.39600816, 0.14368972, 0.7195266,  0.07824122,
        0.00533229, 0.5622921,  0.35415828, 0.11940759, 0.72893023, 0.33766487,
    });
    transposePermutations5->SetExpectedShape({4, 2, 3});
    EXPECT_TRUE(utils::Test(transposePermutations5));
}

TEST_F(TransposeTests, TransposePermutations6) {
    Transpose* transposePermutations6 = new Transpose();
    transposePermutations6->SetInput(
        {2, 3, 4}, {
                       0.7760998,  0.8363521,  0.10145967, 0.00533229, 0.8190919,  0.83241564,
                       0.39479077, 0.5622921,  0.9306249,  0.00480607, 0.39600816, 0.35415828,
                       0.43689877, 0.7603583,  0.14368972, 0.11940759, 0.4834097,  0.6982117,
                       0.7195266,  0.72893023, 0.896649,   0.13060148, 0.07824122, 0.33766487,
                   });
    transposePermutations6->SetPermutation({2, 1, 0});
    transposePermutations6->SetExpectedBuffer({
        0.7760998,  0.43689877, 0.8190919,  0.4834097,  0.9306249,  0.896649,
        0.8363521,  0.7603583,  0.83241564, 0.6982117,  0.00480607, 0.13060148,
        0.10145967, 0.14368972, 0.39479077, 0.7195266,  0.39600816, 0.07824122,
        0.00533229, 0.11940759, 0.5622921,  0.72893023, 0.35415828, 0.33766487,
    });
    transposePermutations6->SetExpectedShape({4, 3, 2});
    EXPECT_TRUE(utils::Test(transposePermutations6));
}