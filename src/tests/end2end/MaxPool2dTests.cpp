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
    void SetPadding(std::vector<int32_t> padding) {
        mPadding = std::move(padding);
        mOptions.padding = mPadding.data();
    }
    void SetWindowDimensions(std::vector<int32_t> dimensions) {
        mWindowDimensions = std::move(dimensions);
        mOptions.windowDimensions = mWindowDimensions.data();
    }

  private:
    webnn::Pool2dOptions mOptions;
    std::vector<int32_t> mPadding;
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