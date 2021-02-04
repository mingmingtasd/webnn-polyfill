// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "utils.h"

namespace InferenceEngine {

namespace {

float asfloat(uint32_t v) {
  union {
    float f;
    std::uint32_t u;
  } converter = {0};
  converter.u = v;
  return converter.f;
}

}  // namespace

short f32tof16(float x) {
  static float min16 = asfloat((127 - 14) << 23);

  static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
  static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

  static constexpr std::uint32_t EXP_MASK_F32 = 0x7F800000U;

  union {
    float f;
    uint32_t u;
  } v = {0};
  v.f = x;

  uint32_t s = (v.u >> 16) & 0x8000;

  v.u &= 0x7FFFFFFF;

  if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
    if (v.u & 0x007FFFFF) {
      return static_cast<short>(s | (v.u >> (23 - 10)) | 0x0200);
    } else {
      return static_cast<short>(s | (v.u >> (23 - 10)));
    }
  }

  float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
  v.f += halfULP;

  if (v.f < min16 * 0.5f) {
    return static_cast<short>(s);
  }

  if (v.f < min16) {
    return static_cast<short>(s | (1 << 10));
  }

  if (v.f >= max16) {
    return static_cast<short>(max16f16 | s);
  }

  v.u -= ((127 - 15) << 23);

  v.u >>= (23 - 10);

  return static_cast<short>(v.u | s);
}

}  // namespace InferenceEngine
