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

class Conv2dTests : public testing::Test {};

class Conv2d : public utils::WrappedModel {
  public:
    Conv2d() : mOptions({}) {
    }
    webnn::Operand GenerateOutput(webnn::ModelBuilder nn) override {
        webnn::Operand input = nn.Input("input", InputDesc());
        webnn::Operand constant = nn.Constant(ConstantDesc(), ConstantBuffer(), ConstantLength());
        return nn.Conv2d(input, constant, &mOptions);
    }
    void SetPadding(std::vector<int32_t> padding) {
        mPadding = std::move(padding);
        mOptions.padding = mPadding.data();
    }

  private:
    webnn::Conv2dOptions mOptions;
    std::vector<int32_t> mPadding;
};

TEST_F(Conv2dTests, Conv2d) {
    Conv2d* conv2d = new Conv2d();
    conv2d->SetInput({1, 1, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    conv2d->SetConstant({1, 1, 3, 3}, std::vector<float>(9, 1));
    conv2d->SetPadding({1, 1, 1, 1});
    conv2d->SetExpectedBuffer({12.,  21.,  27., 33.,  24.,  33.,  54., 63.,  72.,
                               51.,  63.,  99., 108., 117., 81.,  93., 144., 153.,
                               162., 111., 72., 111., 117., 123., 84.});
    conv2d->SetExpectedShape({1, 1, 5, 5});
    EXPECT_TRUE(utils::Test(conv2d));
}