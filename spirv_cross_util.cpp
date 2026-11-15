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

#include "spirv_cross_util.hpp"
#include "spirv_common.hpp"

using namespace SPIRV_CROSS_SPV_HEADER_NAMESPACE;
using namespace SPIRV_CROSS_NAMESPACE;

namespace spirv_cross_util
{
namespace
{
struct BindingPoint
{
	uint32_t descriptor_set = 0;
	uint32_t binding = 0;
	bool has_descriptor_set = false;
	bool has_binding = false;
};

BindingPoint get_binding_point(const Compiler &compiler, VariableID id)
{
	BindingPoint point;
	point.has_descriptor_set = compiler.has_decoration(id, DecorationDescriptorSet);
	point.has_binding = compiler.has_decoration(id, DecorationBinding);

	if (point.has_descriptor_set)
		point.descriptor_set = compiler.get_decoration(id, DecorationDescriptorSet);
	if (point.has_binding)
		point.binding = compiler.get_decoration(id, DecorationBinding);

	return point;
}

void inherit_missing_binding_components(BindingPoint &dst, const BindingPoint &src)
{
	if (!dst.has_descriptor_set && src.has_descriptor_set)
	{
		dst.descriptor_set = src.descriptor_set;
		dst.has_descriptor_set = true;
	}

	if (!dst.has_binding && src.has_binding)
	{
		dst.binding = src.binding;
		dst.has_binding = true;
	}
}

std::string printable_name(const Compiler &compiler, VariableID id)
{
	auto name = compiler.get_name(id);
	return name.empty() ? join("ID ", id) : name;
}

std::string describe_pair(const Compiler &compiler, const CombinedImageSampler &sampler)
{
	return join("image \"", printable_name(compiler, sampler.image_id), "\" + sampler \"",
	            printable_name(compiler, sampler.sampler_id), "\"");
}
} // namespace

void rename_interface_variable(Compiler &compiler, const SmallVector<Resource> &resources, uint32_t location,
                               const std::string &name)
{
	for (auto &v : resources)
	{
		if (!compiler.has_decoration(v.id, DecorationLocation))
			continue;

		auto loc = compiler.get_decoration(v.id, DecorationLocation);
		if (loc != location)
			continue;

		auto &type = compiler.get_type(v.base_type_id);

		// This is more of a friendly variant. If we need to rename interface variables, we might have to rename
		// structs as well and make sure all the names match up.
		if (type.basetype == SPIRType::Struct)
		{
			compiler.set_name(v.base_type_id, join("SPIRV_Cross_Interface_Location", location));
			for (uint32_t i = 0; i < uint32_t(type.member_types.size()); i++)
				compiler.set_member_name(v.base_type_id, i, join("InterfaceMember", i));
		}

		compiler.set_name(v.id, name);
	}
}

SmallVector<CombinedImageSamplerBinding> resolve_combined_image_sampler_bindings(Compiler &compiler)
{
	auto &samplers = compiler.get_combined_image_samplers();
	SmallVector<CombinedImageSamplerBinding> resolved;
	resolved.reserve(samplers.size());

	for (auto &sampler : samplers)
	{
		BindingPoint point = get_binding_point(compiler, sampler.image_id);
		inherit_missing_binding_components(point, get_binding_point(compiler, sampler.sampler_id));
		inherit_missing_binding_components(point, get_binding_point(compiler, sampler.combined_id));

		CombinedImageSamplerBinding binding;
		binding.combined_id = sampler.combined_id;
		binding.image_id = sampler.image_id;
		binding.sampler_id = sampler.sampler_id;
		binding.descriptor_set = point.descriptor_set;
		binding.binding = point.binding;
		binding.has_descriptor_set = point.has_descriptor_set;
		binding.has_binding = point.has_binding;
		resolved.push_back(binding);
	}

	// Resolve everything before mutating the compiler, so a conflict is reported atomically.
	for (size_t i = 0; i < resolved.size(); i++)
	{
		if (!resolved[i].has_binding)
			continue;

		uint32_t set_i = resolved[i].has_descriptor_set ? resolved[i].descriptor_set : 0;
		for (size_t j = 0; j < i; j++)
		{
			if (!resolved[j].has_binding)
				continue;

			uint32_t set_j = resolved[j].has_descriptor_set ? resolved[j].descriptor_set : 0;
			if (set_i == set_j && resolved[i].binding == resolved[j].binding)
			{
				auto &a = samplers[j];
				auto &b = samplers[i];
				SPIRV_CROSS_THROW(join("Combined image sampler binding conflict: ", describe_pair(compiler, a),
				                       " and ", describe_pair(compiler, b), " both resolve to set ", set_i,
				                       ", binding ", resolved[i].binding, "."));
			}
		}
	}

	for (auto &binding : resolved)
	{
		if (binding.has_descriptor_set)
			compiler.set_decoration(binding.combined_id, DecorationDescriptorSet, binding.descriptor_set);
		if (binding.has_binding)
			compiler.set_decoration(binding.combined_id, DecorationBinding, binding.binding);
	}

	return resolved;
}

void inherit_combined_sampler_bindings(Compiler &compiler)
{
	resolve_combined_image_sampler_bindings(compiler);
}
} // namespace spirv_cross_util
