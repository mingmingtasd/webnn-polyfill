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

class MaxPool2dTests : public testing::Test {};

class MaxPool2d : public utils::WrappedModel {
  public:
    MaxPool2d() : mOptions({}) {
    }
    webnn::Operand GenerateOutput(webnn::ModelBuilder nn) override {
        webnn::Operand input = nn.Input("input", InputDesc());
        return nn.MaxPool2d(input, &mOptions);
    }
    void SetDilations(std::vector<int32_t> dilations) {
        mDilations = std::move(dilations);
        mOptions.dilations = mDilations.data();
        mOptions.dilationsCount = mDilations.size();
    }
    void SetPadding(std::vector<int32_t> padding) {
        mPadding = std::move(padding);
        mOptions.padding = mPadding.data();
    }
    void SetStrides(std::vector<int32_t> strides) {
        mStrides = std::move(strides);
        mOptions.strides = mStrides.data();
        mOptions.stridesCount = mStrides.size();
    }
    void SetWindowDimensions(std::vector<int32_t> dimensions) {
        mWindowDimensions = std::move(dimensions);
        mOptions.windowDimensions = mWindowDimensions.data();
    }

  private:
    std::vector<int32_t> mDilations;
    webnn::Pool2dOptions mOptions;
    std::vector<int32_t> mPadding;
    std::vector<int32_t> mStrides;
    std::vector<int32_t> mWindowDimensions;
};

TEST_F(MaxPool2dTests, MaxPool2d) {
    MaxPool2d* maxPool2d = new MaxPool2d();
    maxPool2d->SetInput({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    maxPool2d->SetWindowDimensions({3, 3});
    maxPool2d->SetExpectedBuffer({11, 12, 15, 16});
    maxPool2d->SetExpectedShape({1, 1, 2, 2});
    EXPECT_TRUE(utils::Test(maxPool2d));
}

TEST_F(MaxPool2dTests, MaxPool2dDilations) {
    MaxPool2d* maxPool2dDilations = new MaxPool2d();
    maxPool2dDilations->SetInput({1, 1, 4, 4},
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    maxPool2dDilations->SetDilations({2, 2});
    maxPool2dDilations->SetWindowDimensions({2, 2});
    maxPool2dDilations->SetExpectedBuffer({11, 12, 15, 16});
    maxPool2dDilations->SetExpectedShape({1, 1, 2, 2});
    EXPECT_TRUE(utils::Test(maxPool2dDilations));
}

TEST_F(MaxPool2dTests, MaxPool2dPads) {
    MaxPool2d* maxPool2dPads = new MaxPool2d();
    maxPool2dPads->SetInput({1, 1, 5, 5}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
    maxPool2dPads->SetPadding({2, 2, 2, 2});
    maxPool2dPads->SetWindowDimensions({5, 5});
    maxPool2dPads->SetExpectedBuffer({
        13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
        25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25,
    });
    maxPool2dPads->SetExpectedShape({1, 1, 5, 5});
    EXPECT_TRUE(utils::Test(maxPool2dPads));
}

TEST_F(MaxPool2dTests, MaxPool2dStrides) {
    MaxPool2d* maxPool2dStrides = new MaxPool2d();
    maxPool2dStrides->SetInput({1, 1, 5, 5}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
    maxPool2dStrides->SetStrides({2, 2});
    maxPool2dStrides->SetWindowDimensions({2, 2});
    maxPool2dStrides->SetExpectedBuffer({7, 9, 17, 19});
    maxPool2dStrides->SetExpectedShape({1, 1, 2, 2});
    EXPECT_TRUE(utils::Test(maxPool2dStrides));
}