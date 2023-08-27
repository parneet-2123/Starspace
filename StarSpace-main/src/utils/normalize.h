/**
 * Copyright (c) funcoding, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace spacexplore {

// In-place normalization of UTF-8 strings.
extern void normalize_text(std::string& buf);

}
