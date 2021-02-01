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

class ReshapeTests : public testing::Test {};

class ReshapeModel : public utils::WrappedModel {
  public:
    ReshapeModel() = default;
    webnn::Operand GenerateOutput(webnn::ModelBuilder nn) override {
        webnn::Operand input = nn.Input("input", InputDesc());
        return nn.Reshape(input, mNewShape.data(), mNewShape.size());
    }
    void SetNewShape(std::vector<int32_t> newshape) {
        mNewShape = std::move(newshape);
    }

  private:
    std::vector<int32_t> mNewShape;
};

TEST_F(ReshapeTests, ReshapeReorderedAllDims) {
    ReshapeModel* reshapeReorderedAllDims = new ReshapeModel();
    reshapeReorderedAllDims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReorderedAllDims->SetNewShape({4, 2, 3});
    reshapeReorderedAllDims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReorderedAllDims->SetExpectedShape({4, 2, 3});
    EXPECT_TRUE(utils::Test(reshapeReorderedAllDims));
}

TEST_F(ReshapeTests, ReshapeReorderedLastDims) {
    ReshapeModel* reshapeReorderedLastDims = new ReshapeModel();
    reshapeReorderedLastDims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReorderedLastDims->SetNewShape({2, 4, 3});
    reshapeReorderedLastDims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReorderedLastDims->SetExpectedShape({2, 4, 3});
    EXPECT_TRUE(utils::Test(reshapeReorderedLastDims));
}

TEST_F(ReshapeTests, ReshapeReducedDims) {
    ReshapeModel* reshapeReducedDims = new ReshapeModel();
    reshapeReducedDims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReducedDims->SetNewShape({2, 12});
    reshapeReducedDims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeReducedDims->SetExpectedShape({2, 12});
    EXPECT_TRUE(utils::Test(reshapeReducedDims));
}

TEST_F(ReshapeTests, ReshapeExtendedDims) {
    ReshapeModel* reshapeExtendedDims = new ReshapeModel();
    reshapeExtendedDims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeExtendedDims->SetNewShape({2, 3, 2, 2});
    reshapeExtendedDims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeExtendedDims->SetExpectedShape({2, 3, 2, 2});
    EXPECT_TRUE(utils::Test(reshapeExtendedDims));
}

TEST_F(ReshapeTests, ReshapeOneDim) {
    ReshapeModel* reshapeOneDim = new ReshapeModel();
    reshapeOneDim->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeOneDim->SetNewShape({24});
    reshapeOneDim->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeOneDim->SetExpectedShape({24});
    EXPECT_TRUE(utils::Test(reshapeOneDim));
}

TEST_F(ReshapeTests, ReshapeNegativeDim) {
    ReshapeModel* reshapeNegativeDim = new ReshapeModel();
    reshapeNegativeDim->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeNegativeDim->SetNewShape({2, -1, 2});
    reshapeNegativeDim->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeNegativeDim->SetExpectedShape({2, 6, 2});
    EXPECT_TRUE(utils::Test(reshapeNegativeDim));
}

TEST_F(ReshapeTests, ReshapeNegativeDim1) {
    ReshapeModel* reshapeNegativeDim1 = new ReshapeModel();
    reshapeNegativeDim1->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeNegativeDim1->SetNewShape({-1, 2, 3, 4});
    reshapeNegativeDim1->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshapeNegativeDim1->SetExpectedShape({1, 2, 3, 4});
    EXPECT_TRUE(utils::Test(reshapeNegativeDim1));
}