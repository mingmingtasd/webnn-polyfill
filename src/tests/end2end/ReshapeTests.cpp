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
    wnn::Operand GenerateOutput(wnn::ModelBuilder nn) override {
        wnn::Operand input = nn.Input("input", InputDesc());
        return nn.Reshape(input, new_shape_.data(), new_shape_.size());
    }
    void SetNewShape(std::vector<int32_t> new_shape) {
        new_shape_ = std::move(new_shape);
    }

  private:
    std::vector<int32_t> new_shape_;
};

TEST_F(ReshapeTests, reshape_reordered_all_dims) {
    ReshapeModel* reshape_reordered_all_dims = new ReshapeModel();
    reshape_reordered_all_dims->SetInput(
        {2, 3, 4},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reordered_all_dims->SetNewShape({4, 2, 3});
    reshape_reordered_all_dims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reordered_all_dims->SetExpectedShape({4, 2, 3});
    EXPECT_TRUE(utils::Test(reshape_reordered_all_dims));
}

TEST_F(ReshapeTests, reshape_reordered_last_dims) {
    ReshapeModel* reshape_reordered_last_dims = new ReshapeModel();
    reshape_reordered_last_dims->SetInput(
        {2, 3, 4},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reordered_last_dims->SetNewShape({2, 4, 3});
    reshape_reordered_last_dims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reordered_last_dims->SetExpectedShape({2, 4, 3});
    EXPECT_TRUE(utils::Test(reshape_reordered_last_dims));
}

TEST_F(ReshapeTests, reshape_reduced_dims) {
    ReshapeModel* reshape_reduced_dims = new ReshapeModel();
    reshape_reduced_dims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reduced_dims->SetNewShape({2, 12});
    reshape_reduced_dims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_reduced_dims->SetExpectedShape({2, 12});
    EXPECT_TRUE(utils::Test(reshape_reduced_dims));
}

TEST_F(ReshapeTests, reshape_extended_dims) {
    ReshapeModel* reshape_extended_dims = new ReshapeModel();
    reshape_extended_dims->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_extended_dims->SetNewShape({2, 3, 2, 2});
    reshape_extended_dims->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_extended_dims->SetExpectedShape({2, 3, 2, 2});
    EXPECT_TRUE(utils::Test(reshape_extended_dims));
}

TEST_F(ReshapeTests, reshape_one_dim) {
    ReshapeModel* reshape_one_dim = new ReshapeModel();
    reshape_one_dim->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_one_dim->SetNewShape({24});
    reshape_one_dim->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_one_dim->SetExpectedShape({24});
    EXPECT_TRUE(utils::Test(reshape_one_dim));
}

TEST_F(ReshapeTests, reshape_negative_dim) {
    ReshapeModel* reshape_negative_dim = new ReshapeModel();
    reshape_negative_dim->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_negative_dim->SetNewShape({2, -1, 2});
    reshape_negative_dim->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_negative_dim->SetExpectedShape({2, 6, 2});
    EXPECT_TRUE(utils::Test(reshape_negative_dim));
}

TEST_F(ReshapeTests, reshape_negative_dim_1) {
    ReshapeModel* reshape_negative_dim_1 = new ReshapeModel();
    reshape_negative_dim_1->SetInput({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_negative_dim_1->SetNewShape({-1, 2, 3, 4});
    reshape_negative_dim_1->SetExpectedBuffer(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    reshape_negative_dim_1->SetExpectedShape({1, 2, 3, 4});
    EXPECT_TRUE(utils::Test(reshape_negative_dim_1));
}