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
    AveragePool2d() : mOptions({}) {
    }
    webnn::Operand GenerateOutput(webnn::ModelBuilder nn) override {
        webnn::Operand input = nn.Input("input", InputDesc());
        return nn.AveragePool2d(input, &mOptions);
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
    webnn::Pool2dOptions mOptions;
    std::vector<int32_t> mPadding;
    std::vector<int32_t> mStrides;
    std::vector<int32_t> mWindowDimensions;
};

TEST_F(AveragePool2dTests, AveragePool2d) {
    AveragePool2d* averagePool2d = new AveragePool2d();
    averagePool2d->SetInput({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    averagePool2d->SetWindowDimensions({3, 3});
    averagePool2d->SetExpectedBuffer({6, 7, 10, 11});
    averagePool2d->SetExpectedShape({1, 1, 2, 2});
    EXPECT_TRUE(utils::Test(averagePool2d));
}

TEST_F(AveragePool2dTests, AveragePool2dPads) {
    AveragePool2d* averagePool2dPads = new AveragePool2d();
    averagePool2dPads->SetInput({1, 1, 5, 5}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
    averagePool2dPads->SetPadding({2, 2, 2, 2});
    averagePool2dPads->SetWindowDimensions({5, 5});
    averagePool2dPads->SetExpectedBuffer({7,    7.5,  8,    8.5,  9,    9.5,  10,   10.5, 11,
                                          11.5, 12,   12.5, 13,   13.5, 14,   14.5, 15,   15.5,
                                          16,   16.5, 17,   17.5, 18,   18.5, 19});
    averagePool2dPads->SetExpectedShape({1, 1, 5, 5});
    EXPECT_TRUE(utils::Test(averagePool2dPads));
}

TEST_F(AveragePool2dTests, AveragePool2dStrides) {
    AveragePool2d* averagePool2dStrides = new AveragePool2d();
    averagePool2dStrides->SetInput({1, 1, 5, 5},
                                   {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
    averagePool2dStrides->SetWindowDimensions({2, 2});
    averagePool2dStrides->SetStrides({2, 2});
    averagePool2dStrides->SetExpectedBuffer({4, 6, 14, 16});
    averagePool2dStrides->SetExpectedShape({1, 1, 2, 2});
    EXPECT_TRUE(utils::Test(averagePool2dStrides));
}

TEST_F(AveragePool2dTests, GlobalAveragePool2d) {
    AveragePool2d* globalAveragePool2d = new AveragePool2d();
    globalAveragePool2d->SetInput(
        {1, 3, 5, 5},
        {
            -1.1289884,  0.34016284,  0.497431,   2.1915932,   0.42038894,  -0.18261199,
            -0.15769927, -0.26465914, 0.03877424, 0.39492005,  -0.33410737, 0.74918455,
            -1.3542547,  -0.0222946,  0.7094626,  -0.09399617, 0.790736,    -0.75826526,
            0.27656242,  0.46543223,  -1.2342638, 1.1549494,   0.24823844,  0.75670505,
            -1.7108902,  -1.4767597,  -1.4969662, -0.31936142, 0.5327554,   -0.06070877,
            0.31212643,  2.2274113,   1.2775147,  0.59886885,  -1.5765078,  0.18522178,
            0.22655599,  0.88869494,  0.38609484, -0.05860576, -0.72732115, -0.0046324,
            -1.3593693,  -0.6295078,  1.384531,   0.06825881,  0.19907428,  0.20298219,
            -0.8399954,  1.3583295,   0.02117888, -1.0636739,  -0.30460566, -0.92678875,
            -0.09120782, -0.88333017, -0.9641269, 0.6065926,   -0.5830042,  -0.81138134,
            1.3569402,   1.2891295,   0.2508177,  0.20211531,  0.8832168,   -0.19886094,
            -0.61088,    0.682026,    -0.5253442, 1.5022339,   1.0256356,   1.0642492,
            -0.4169051,  -0.8740329,  1.1494869,
        });
    globalAveragePool2d->SetExpectedBuffer({0.07170041, 0.05194739, 0.07117923});
    globalAveragePool2d->SetExpectedShape({1, 3, 1, 1});
    EXPECT_TRUE(utils::Test(globalAveragePool2d));
}