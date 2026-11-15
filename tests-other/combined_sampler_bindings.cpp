#include "spirv_cross.hpp"
#include "spirv_cross_util.hpp"
#include "spirv_glsl.hpp"
#include "spirv_parser.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace spirv_cross;
namespace util = spirv_cross_util;

// Two textures and two samplers, all with distinct binding points.
static const uint32_t merge_spirv[] = {
	0x07230203u, 0x00010300u, 0x0008000bu, 0x0000002bu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
	0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
	0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000001du, 0x00000021u, 0x00030010u,
	0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
	0x00000000u, 0x00080005u, 0x00000012u, 0x706d6173u, 0x705f656cu, 0x28726961u, 0x3b313274u, 0x763b3170u,
	0x003b3266u, 0x00030005u, 0x0000000fu, 0x00000074u, 0x00030005u, 0x00000010u, 0x00000073u, 0x00030005u,
	0x00000011u, 0x00007675u, 0x00050005u, 0x0000001du, 0x67617246u, 0x6f6c6f43u, 0x00000072u, 0x00040005u,
	0x0000001eu, 0x78655475u, 0x00000041u, 0x00050005u, 0x0000001fu, 0x6d615375u, 0x72656c70u, 0x00000030u,
	0x00030005u, 0x00000021u, 0x00565576u, 0x00040005u, 0x00000022u, 0x61726170u, 0x0000006du, 0x00040005u,
	0x00000025u, 0x78655475u, 0x00000042u, 0x00050005u, 0x00000026u, 0x6d615375u, 0x72656c70u, 0x00000031u,
	0x00040005u, 0x00000027u, 0x61726170u, 0x0000006du, 0x00040047u, 0x0000001du, 0x0000001eu, 0x00000000u,
	0x00040047u, 0x0000001eu, 0x00000021u, 0x00000005u, 0x00040047u, 0x0000001eu, 0x00000022u, 0x00000002u,
	0x00040047u, 0x0000001fu, 0x00000021u, 0x00000000u, 0x00040047u, 0x0000001fu, 0x00000022u, 0x00000000u,
	0x00040047u, 0x00000021u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000025u, 0x00000021u, 0x00000009u,
	0x00040047u, 0x00000025u, 0x00000022u, 0x00000003u, 0x00040047u, 0x00000026u, 0x00000021u, 0x00000001u,
	0x00040047u, 0x00000026u, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
	0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00090019u, 0x00000007u, 0x00000006u, 0x00000001u,
	0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u, 0x00000008u, 0x00000000u,
	0x00000007u, 0x0002001au, 0x00000009u, 0x00040020u, 0x0000000au, 0x00000000u, 0x00000009u, 0x00040017u,
	0x0000000bu, 0x00000006u, 0x00000002u, 0x00040020u, 0x0000000cu, 0x00000007u, 0x0000000bu, 0x00040017u,
	0x0000000du, 0x00000006u, 0x00000004u, 0x00060021u, 0x0000000eu, 0x0000000du, 0x00000008u, 0x0000000au,
	0x0000000cu, 0x0003001bu, 0x00000016u, 0x00000007u, 0x00040020u, 0x0000001cu, 0x00000003u, 0x0000000du,
	0x0004003bu, 0x0000001cu, 0x0000001du, 0x00000003u, 0x0004003bu, 0x00000008u, 0x0000001eu, 0x00000000u,
	0x0004003bu, 0x0000000au, 0x0000001fu, 0x00000000u, 0x00040020u, 0x00000020u, 0x00000001u, 0x0000000bu,
	0x0004003bu, 0x00000020u, 0x00000021u, 0x00000001u, 0x0004003bu, 0x00000008u, 0x00000025u, 0x00000000u,
	0x0004003bu, 0x0000000au, 0x00000026u, 0x00000000u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
	0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000000cu, 0x00000022u, 0x00000007u, 0x0004003bu,
	0x0000000cu, 0x00000027u, 0x00000007u, 0x0004003du, 0x0000000bu, 0x00000023u, 0x00000021u, 0x0003003eu,
	0x00000022u, 0x00000023u, 0x00070039u, 0x0000000du, 0x00000024u, 0x00000012u, 0x0000001eu, 0x0000001fu,
	0x00000022u, 0x0004003du, 0x0000000bu, 0x00000028u, 0x00000021u, 0x0003003eu, 0x00000027u, 0x00000028u,
	0x00070039u, 0x0000000du, 0x00000029u, 0x00000012u, 0x00000025u, 0x00000026u, 0x00000027u, 0x00050081u,
	0x0000000du, 0x0000002au, 0x00000024u, 0x00000029u, 0x0003003eu, 0x0000001du, 0x0000002au, 0x000100fdu,
	0x00010038u, 0x00050036u, 0x0000000du, 0x00000012u, 0x00000000u, 0x0000000eu, 0x00030037u, 0x00000008u,
	0x0000000fu, 0x00030037u, 0x0000000au, 0x00000010u, 0x00030037u, 0x0000000cu, 0x00000011u, 0x000200f8u,
	0x00000013u, 0x0004003du, 0x00000007u, 0x00000014u, 0x0000000fu, 0x0004003du, 0x00000009u, 0x00000015u,
	0x00000010u, 0x00050056u, 0x00000016u, 0x00000017u, 0x00000014u, 0x00000015u, 0x0004003du, 0x0000000bu,
	0x00000018u, 0x00000011u, 0x00050057u, 0x0000000du, 0x00000019u, 0x00000017u, 0x00000018u, 0x000200feu,
	0x00000019u, 0x00010038u,
};

// One texture is used with two samplers. Both pairs inherit texture binding 5 set 0.
static const uint32_t conflict_spirv[] = {
	0x07230203u, 0x00010300u, 0x0008000bu, 0x00000020u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
	0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
	0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000016u, 0x00030010u,
	0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
	0x00000000u, 0x00050005u, 0x00000009u, 0x67617246u, 0x6f6c6f43u, 0x00000072u, 0x00040005u, 0x0000000cu,
	0x78655475u, 0x00000000u, 0x00050005u, 0x00000010u, 0x6d615375u, 0x72656c70u, 0x00000030u, 0x00030005u,
	0x00000016u, 0x00565576u, 0x00050005u, 0x0000001au, 0x6d615375u, 0x72656c70u, 0x00000031u, 0x00040047u,
	0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000021u, 0x00000005u, 0x00040047u,
	0x0000000cu, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000010u, 0x00000021u, 0x00000000u, 0x00040047u,
	0x00000010u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000016u, 0x0000001eu, 0x00000000u, 0x00040047u,
	0x0000001au, 0x00000021u, 0x00000001u, 0x00040047u, 0x0000001au, 0x00000022u, 0x00000000u, 0x00020013u,
	0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u,
	0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
	0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u,
	0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u, 0x0000000bu, 0x00000000u, 0x0000000au,
	0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x0002001au, 0x0000000eu, 0x00040020u, 0x0000000fu,
	0x00000000u, 0x0000000eu, 0x0004003bu, 0x0000000fu, 0x00000010u, 0x00000000u, 0x0003001bu, 0x00000012u,
	0x0000000au, 0x00040017u, 0x00000014u, 0x00000006u, 0x00000002u, 0x00040020u, 0x00000015u, 0x00000001u,
	0x00000014u, 0x0004003bu, 0x00000015u, 0x00000016u, 0x00000001u, 0x0004003bu, 0x0000000fu, 0x0000001au,
	0x00000000u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
	0x0004003du, 0x0000000au, 0x0000000du, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x00000011u, 0x00000010u,
	0x00050056u, 0x00000012u, 0x00000013u, 0x0000000du, 0x00000011u, 0x0004003du, 0x00000014u, 0x00000017u,
	0x00000016u, 0x00050057u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u, 0x0004003du, 0x0000000au,
	0x00000019u, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x0000001bu, 0x0000001au, 0x00050056u, 0x00000012u,
	0x0000001cu, 0x00000019u, 0x0000001bu, 0x0004003du, 0x00000014u, 0x0000001du, 0x00000016u, 0x00050057u,
	0x00000007u, 0x0000001eu, 0x0000001cu, 0x0000001du, 0x00050081u, 0x00000007u, 0x0000001fu, 0x00000018u,
	0x0000001eu, 0x0003003eu, 0x00000009u, 0x0000001fu, 0x000100fdu, 0x00010038u,
};

// No descriptor set decorations. The image binding is removed after building the combined samplers.
static const uint32_t fallback_spirv[] = {
	0x07230203u, 0x00010300u, 0x0008000bu, 0x00000019u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
	0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
	0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000016u, 0x00030010u,
	0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
	0x00000000u, 0x00050005u, 0x00000009u, 0x67617246u, 0x6f6c6f43u, 0x00000072u, 0x00040005u, 0x0000000cu,
	0x78655475u, 0x00000000u, 0x00050005u, 0x00000010u, 0x6d615375u, 0x72656c70u, 0x00000000u, 0x00030005u,
	0x00000016u, 0x00565576u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000cu,
	0x00000021u, 0x00000005u, 0x00040047u, 0x0000000cu, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000010u,
	0x00000021u, 0x0000000bu, 0x00040047u, 0x00000010u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000016u,
	0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
	0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
	0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au,
	0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u,
	0x0000000bu, 0x00000000u, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x0002001au,
	0x0000000eu, 0x00040020u, 0x0000000fu, 0x00000000u, 0x0000000eu, 0x0004003bu, 0x0000000fu, 0x00000010u,
	0x00000000u, 0x0003001bu, 0x00000012u, 0x0000000au, 0x00040017u, 0x00000014u, 0x00000006u, 0x00000002u,
	0x00040020u, 0x00000015u, 0x00000001u, 0x00000014u, 0x0004003bu, 0x00000015u, 0x00000016u, 0x00000001u,
	0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
	0x0000000au, 0x0000000du, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x00000011u, 0x00000010u, 0x00050056u,
	0x00000012u, 0x00000013u, 0x0000000du, 0x00000011u, 0x0004003du, 0x00000014u, 0x00000017u, 0x00000016u,
	0x00050057u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u,
	0x000100fdu, 0x00010038u,
};

#define CHECK(x) \
	do \
	{ \
		if (!(x)) \
		{ \
			fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
			return false; \
		} \
	} while (0)

static std::unique_ptr<CompilerGLSL> make_compiler(const uint32_t *words, size_t count)
{
	std::vector<uint32_t> spirv(words, words + count);
	Parser parser(std::move(spirv));
	parser.parse();
	return std::unique_ptr<CompilerGLSL>(new CompilerGLSL(std::move(parser.get_parsed_ir())));
}

static const Resource *find_resource(Compiler &compiler, const SmallVector<Resource> &resources, const char *name)
{
	for (auto &resource : resources)
		if (compiler.get_name(resource.id) == name)
			return &resource;
	return nullptr;
}

static const util::CombinedImageSamplerBinding *
find_binding(const SmallVector<util::CombinedImageSamplerBinding> &bindings, const char *image_name,
             Compiler &compiler)
{
	for (auto &binding : bindings)
		if (compiler.get_name(binding.image_id) == image_name)
			return &binding;
	return nullptr;
}

static bool contains(const std::string &haystack, const char *needle)
{
	return haystack.find(needle) != std::string::npos;
}

static uint64_t fnv1a(const std::string &text)
{
	uint64_t hash = 1469598103934665603ull;
	for (unsigned char c : text)
	{
		hash ^= c;
		hash *= 1099511628211ull;
	}
	return hash;
}

static std::string compile_vulkan_glsl(CompilerGLSL &compiler)
{
	auto options = compiler.get_common_options();
	options.vulkan_semantics = true;
	compiler.set_common_options(options);
	return compiler.compile();
}

static bool test_inherited_binding_points()
{
	auto compiler = make_compiler(merge_spirv, sizeof(merge_spirv) / sizeof(merge_spirv[0]));
	compiler->build_combined_image_samplers();
	CHECK(compiler->get_combined_image_samplers().size() == 2);

	auto bindings = util::resolve_combined_image_sampler_bindings(*compiler);
	CHECK(bindings.size() == 2);

	auto *a = find_binding(bindings, "uTexA", *compiler);
	auto *b = find_binding(bindings, "uTexB", *compiler);
	CHECK(a && b);
	CHECK(compiler->get_name(a->sampler_id) == "uSampler0");
	CHECK(compiler->get_name(b->sampler_id) == "uSampler1");
	CHECK(a->has_descriptor_set && a->descriptor_set == 2);
	CHECK(a->has_binding && a->binding == 5);
	CHECK(b->has_descriptor_set && b->descriptor_set == 3);
	CHECK(b->has_binding && b->binding == 9);
	CHECK(compiler->get_decoration(a->combined_id, DecorationDescriptorSet) == 2);
	CHECK(compiler->get_decoration(a->combined_id, DecorationBinding) == 5);
	CHECK(compiler->get_decoration(b->combined_id, DecorationDescriptorSet) == 3);
	CHECK(compiler->get_decoration(b->combined_id, DecorationBinding) == 9);

	auto glsl = compile_vulkan_glsl(*compiler);
	CHECK(contains(glsl, "layout(set = 2, binding = 5)"));
	CHECK(contains(glsl, "layout(set = 3, binding = 9)"));
	return true;
}

static bool test_sampler_fallback()
{
	auto compiler = make_compiler(fallback_spirv, sizeof(fallback_spirv) / sizeof(fallback_spirv[0]));
	auto resources = compiler->get_shader_resources();
	auto *image = find_resource(*compiler, resources.separate_images, "uTex");
	CHECK(image);

	compiler->build_combined_image_samplers();
	compiler->unset_decoration(image->id, DecorationDescriptorSet);
	compiler->unset_decoration(image->id, DecorationBinding);

	auto bindings = util::resolve_combined_image_sampler_bindings(*compiler);
	CHECK(bindings.size() == 1);
	CHECK(bindings[0].has_descriptor_set && bindings[0].descriptor_set == 0);
	CHECK(bindings[0].has_binding && bindings[0].binding == 11);
	CHECK(compiler->get_decoration(bindings[0].combined_id, DecorationDescriptorSet) == 0);
	CHECK(compiler->get_decoration(bindings[0].combined_id, DecorationBinding) == 11);
	return true;
}

static bool test_conflict_is_atomic()
{
	auto compiler = make_compiler(conflict_spirv, sizeof(conflict_spirv) / sizeof(conflict_spirv[0]));
	compiler->build_combined_image_samplers();
	auto combined = compiler->get_combined_image_samplers();
	CHECK(combined.size() == 2);

	bool threw = false;
	try
	{
		util::resolve_combined_image_sampler_bindings(*compiler);
	}
	catch (const CompilerError &e)
	{
		threw = true;
		std::string message = e.what();
		CHECK(contains(message, "binding conflict"));
		CHECK(contains(message, "uTex"));
		CHECK(contains(message, "uSampler0"));
		CHECK(contains(message, "uSampler1"));
		CHECK(contains(message, "set 0, binding 5"));
	}
	CHECK(threw);

	for (auto &sampler : combined)
	{
		CHECK(!compiler->has_decoration(sampler.combined_id, DecorationDescriptorSet));
		CHECK(!compiler->has_decoration(sampler.combined_id, DecorationBinding));
	}
	return true;
}

static bool test_descriptor_set_disambiguates_binding()
{
	auto compiler = make_compiler(merge_spirv, sizeof(merge_spirv) / sizeof(merge_spirv[0]));
	auto resources = compiler->get_shader_resources();
	auto *image = find_resource(*compiler, resources.separate_images, "uTexB");
	CHECK(image);

	compiler->build_combined_image_samplers();
	compiler->set_decoration(image->id, DecorationBinding, 5);
	auto bindings = util::resolve_combined_image_sampler_bindings(*compiler);
	CHECK(bindings.size() == 2);

	auto *a = find_binding(bindings, "uTexA", *compiler);
	auto *b = find_binding(bindings, "uTexB", *compiler);
	CHECK(a && b);
	CHECK(a->descriptor_set == 2 && a->binding == 5);
	CHECK(b->descriptor_set == 3 && b->binding == 5);
	return true;
}

static bool test_missing_binding_components_are_preserved()
{
	auto compiler = make_compiler(fallback_spirv, sizeof(fallback_spirv) / sizeof(fallback_spirv[0]));
	auto resources = compiler->get_shader_resources();
	auto *image = find_resource(*compiler, resources.separate_images, "uTex");
	CHECK(image);

	compiler->build_combined_image_samplers();
	CHECK(compiler->get_combined_image_samplers().size() == 1);
	auto sampler_id = compiler->get_combined_image_samplers()[0].sampler_id;
	compiler->unset_decoration(image->id, DecorationDescriptorSet);
	compiler->unset_decoration(image->id, DecorationBinding);
	compiler->unset_decoration(sampler_id, DecorationDescriptorSet);
	compiler->unset_decoration(sampler_id, DecorationBinding);

	auto bindings = util::resolve_combined_image_sampler_bindings(*compiler);
	CHECK(bindings.size() == 1);
	CHECK(!bindings[0].has_descriptor_set);
	CHECK(!bindings[0].has_binding);
	CHECK(!compiler->has_decoration(bindings[0].combined_id, DecorationDescriptorSet));
	CHECK(!compiler->has_decoration(bindings[0].combined_id, DecorationBinding));
	return true;
}

static bool test_thousand_conversions_are_stable()
{
	auto baseline_compiler = make_compiler(merge_spirv, sizeof(merge_spirv) / sizeof(merge_spirv[0]));
	baseline_compiler->build_combined_image_samplers();
	util::resolve_combined_image_sampler_bindings(*baseline_compiler);
	auto baseline = compile_vulkan_glsl(*baseline_compiler);
	uint64_t baseline_hash = fnv1a(baseline);

	for (unsigned i = 0; i < 1000; i++)
	{
		auto compiler = make_compiler(merge_spirv, sizeof(merge_spirv) / sizeof(merge_spirv[0]));
		compiler->build_combined_image_samplers();
		auto bindings = util::resolve_combined_image_sampler_bindings(*compiler);
		CHECK(bindings.size() == 2);
		auto glsl = compile_vulkan_glsl(*compiler);
		CHECK(fnv1a(glsl) == baseline_hash);
	}

	printf("consistency: 1000/1000 identical, fnv1a=%016llx\n",
	       static_cast<unsigned long long>(baseline_hash));
	return true;
}

int main()
{
	if (!test_inherited_binding_points() || !test_sampler_fallback() || !test_conflict_is_atomic() ||
	    !test_descriptor_set_disambiguates_binding() || !test_missing_binding_components_are_preserved() ||
	    !test_thousand_conversions_are_stable())
		return 1;

	printf("combined sampler binding tests passed\n");
	return 0;
}
