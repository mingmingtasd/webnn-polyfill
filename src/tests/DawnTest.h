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

#ifndef TESTS_DAWNTEST_H_
#define TESTS_DAWNTEST_H_

#include <gtest/gtest.h>

void InitDawnEnd2EndTestEnvironment(int argc, char** argv);

class DawnTestEnvironment : public testing::Environment {
  public:
    DawnTestEnvironment(int argc, char** argv);
    ~DawnTestEnvironment() override;

    void SetUp() override;
    void TearDown() override;
};

class DawnTestBase {
    // friend class DawnPerfTestBase;

  public:
    DawnTestBase() = default;
    virtual ~DawnTestBase();

    void SetUp();
    void TearDown();
};
#endif  // TESTS_DAWNTEST_H_
