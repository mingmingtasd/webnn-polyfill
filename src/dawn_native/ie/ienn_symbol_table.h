// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_IENN_SYMBOL_TABLE_H_
#define SERVICES_ML_IENN_SYMBOL_TABLE_H_

#include "late_binding_symbol_table.h"

namespace dawn_native {

// The ienn symbols we need, as an X-Macro list.
#define IE_SYMBOLS_LIST                                                        \
  X(ie_create_model)                                                           \
  X(ie_model_free)                                                             \
  X(ie_model_add_constant)                                                     \
  X(ie_model_add_input)                                                        \
  X(ie_model_add_output)                                                       \
  X(ie_model_add_mat_mul)                                                      \
  X(ie_operand_free)                                                           \
  X(ie_model_finish)                                                           \
  X(ie_create_compilation)                                                     \
  X(ie_compilation_free)                                                       \
  X(ie_compilation_set_input)                                                  \
  X(ie_compilation_compute)                                                    \
  X(ie_compilation_get_output)                                                 \
  X(ie_model_add_binary)                                                       \
  X(ie_model_add_conv2d)

LATE_BINDING_SYMBOL_TABLE_DECLARE_BEGIN(IESymbolTable)
#define X(sym) LATE_BINDING_SYMBOL_TABLE_DECLARE_ENTRY(IESymbolTable, sym)
IE_SYMBOLS_LIST
#undef X
LATE_BINDING_SYMBOL_TABLE_DECLARE_END(IESymbolTable)

IESymbolTable *GetIESymbolTable();

#if defined(_WIN32) || defined(_WIN64) || defined(__linux__)
#define IE(sym) LATESYM_GET(IESymbolTable, GetIESymbolTable(), sym)
#else
#define IE(sym) sym
#endif

} // namespace dawn_native

#endif // SERVICES_ML_IENN_SYMBOL_TABLE_H_
