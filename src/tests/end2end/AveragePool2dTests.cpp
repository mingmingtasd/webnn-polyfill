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

class AveragePool2dTests : public testing::Test {};

class AveragePool2d : public utils::WrappedModel {
  public:
    AveragePool2d() : options_({}) {
    }
    wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
        wnn::Operand input = nn.Input("input", InputDesc());
        return nn.AveragePool2d(input, &options_);
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

TEST_F(AveragePool2dTests, average_pool2d) {
    AveragePool2d* average_pool2d = new AveragePool2d();
    average_pool2d->SetInput({1, 1, 5, 5}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
    average_pool2d->SetPadding({2, 2, 2, 2});
    average_pool2d->SetWindowDimensions({5, 5});
    average_pool2d->SetExpectedBuffer({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                       11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                       16,   16.5, 17,   17.5, 18,   18.5, 19});
    average_pool2d->SetExpectedShape({1, 1, 5, 5});
    EXPECT_TRUE(utils::Test(average_pool2d));
}