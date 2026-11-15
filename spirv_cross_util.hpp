/*
 * Copyright 2015-2021 Arm Limited
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#ifndef SPIRV_CROSS_UTIL_HPP
#define SPIRV_CROSS_UTIL_HPP

#include "spirv_cross.hpp"

namespace spirv_cross_util
{
void rename_interface_variable(SPIRV_CROSS_NAMESPACE::Compiler &compiler,
                               const SPIRV_CROSS_NAMESPACE::SmallVector<SPIRV_CROSS_NAMESPACE::Resource> &resources,
                               uint32_t location, const std::string &name);

struct CombinedImageSamplerBinding
{
	SPIRV_CROSS_NAMESPACE::VariableID combined_id = 0;
	SPIRV_CROSS_NAMESPACE::VariableID image_id = 0;
	SPIRV_CROSS_NAMESPACE::VariableID sampler_id = 0;
	uint32_t descriptor_set = 0;
	uint32_t binding = 0;
	bool has_descriptor_set = false;
	bool has_binding = false;
};

// Resolves the unified binding point for every combined image sampler created by
// Compiler::build_combined_image_samplers().
//
// The image binding point is authoritative. A separate image and sampler may legitimately use
// different source binding points; after merging, the image point is the descriptor location and
// the sampler becomes state attached to that descriptor. If the image is missing a descriptor set
// or binding, the corresponding component is inherited from the sampler. A binding point already
// present on the combined variable is used as a final fallback.
//
// Throws CompilerError if two distinct image/sampler pairs resolve to the same (set, binding).
// The compiler is left unchanged when this happens.
SPIRV_CROSS_NAMESPACE::SmallVector<CombinedImageSamplerBinding>
resolve_combined_image_sampler_bindings(SPIRV_CROSS_NAMESPACE::Compiler &compiler);

void inherit_combined_sampler_bindings(SPIRV_CROSS_NAMESPACE::Compiler &compiler);
} // namespace spirv_cross_util

#endif
