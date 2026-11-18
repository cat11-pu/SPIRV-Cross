#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "spirv_parser.hpp"
#include "spirv_reflect.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPIRV_CROSS_NAMESPACE;

static std::vector<uint32_t> load_spirv(const char *path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
		throw std::runtime_error("Could not open input file.");

	file.seekg(0, std::ios::end);
	std::streamoff size = file.tellg();
	if (size < 0)
		throw std::runtime_error("Could not determine input size.");
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes(static_cast<size_t>(size));
	if (!bytes.empty() && !file.read(reinterpret_cast<char *>(bytes.data()), size))
		throw std::runtime_error("Could not read input file.");

	// The fuzz corpus intentionally contains byte-truncated words. Zero-fill the
	// final word so the parser, rather than the file loader, decides what is malformed.
	std::vector<uint32_t> spirv((bytes.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0);
	if (!bytes.empty())
		std::memcpy(spirv.data(), bytes.data(), bytes.size());
	return spirv;
}

static std::string one_line(std::string text)
{
	std::replace(text.begin(), text.end(), '\n', ' ');
	std::replace(text.begin(), text.end(), '\r', ' ');
	return text;
}

template <typename CompilerType>
static void compile_one(const ParsedIR &ir, const char *backend)
{
	try
	{
		CompilerType compiler(ir);
		compiler.compile();
		std::cout << "PASS backend=" << backend << '\n';
	}
	catch (const std::exception &e)
	{
		std::cout << "REJECT backend=" << backend << " error=" << one_line(e.what()) << '\n';
	}
	catch (...)
	{
		std::cout << "REJECT backend=" << backend << " error=unknown exception" << '\n';
	}
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " <module.spv>\n";
		return 2;
	}

	try
	{
		Parser parser(load_spirv(argv[1]));
		try
		{
			parser.parse();
		}
		catch (const std::exception &e)
		{
			std::cout << "REJECT backend=parser error=" << one_line(e.what()) << '\n';
			return 0;
		}

		const ParsedIR &ir = parser.get_parsed_ir();
		compile_one<CompilerGLSL>(ir, "glsl");
		compile_one<CompilerHLSL>(ir, "hlsl");
		compile_one<CompilerMSL>(ir, "msl");
		compile_one<CompilerReflection>(ir, "reflect");
	}
	catch (const std::exception &e)
	{
		std::cout << "REJECT backend=runner error=" << one_line(e.what()) << '\n';
	}
	catch (...)
	{
		std::cout << "REJECT backend=runner error=unknown exception\n";
	}

	return 0;
}
