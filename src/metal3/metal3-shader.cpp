#include "metal3-backend.h"
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <limits>
#include <memory>
#include <metal_irconverter/metal_irconverter.h>
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#pragma push_macro("BOOL")
#undef BOOL
#define BOOL MetalDxcBOOL
#include <dxcapi.h>
#pragma push_macro("interface")
#undef interface
#define interface struct
#include <d3d12shader.h>
#pragma pop_macro("interface")
#pragma pop_macro("BOOL")

namespace nvrhi::metal3
{
    static std::string trimCopy(std::string value)
    {
        auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch); });
        auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
        if (first >= last)
            return {};
        return std::string(first, last);
    }

    static bool readTextFile(const std::string& path, std::string& out)
    {
        std::ifstream file(path);
        if (!file)
            return false;

        std::ostringstream ss;
        ss << file.rdbuf();
        out = ss.str();
        return true;
    }

    static bool parseMscArgumentType(const std::string& value, MscArgumentType& out)
    {
        if (value == "SRV")
        {
            out = MscArgumentType::SRV;
            return true;
        }
        if (value == "UAV")
        {
            out = MscArgumentType::UAV;
            return true;
        }
        if (value == "CBV")
        {
            out = MscArgumentType::CBV;
            return true;
        }
        if (value == "Sampler")
        {
            out = MscArgumentType::Sampler;
            return true;
        }

        return false;
    }

    static std::string normalizeMscVertexInputName(const std::string& value)
    {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            if (std::isalnum(ch))
                result.push_back(static_cast<char>(std::tolower(ch)));
        }

        if (!result.empty() && !std::isdigit(static_cast<unsigned char>(result.back())))
            result.push_back('0');

        return result;
    }

    static bool parseMscReflectionJson(const std::string& json, MscShaderReflection& reflection, MTLSize* computeThreads)
    {
        std::smatch match;
        bool hasResourceCount = false;

        // top-level MSC metadata describes the original HLSL stage and whether
        // the generated Metal entry point needs function constants at creation.
        std::regex shaderTypeRegex(R"name("ShaderType"\s*:\s*"([^"]+)")name");
        if (std::regex_search(json, match, shaderTypeRegex))
            reflection.shaderType = match[1].str();

        std::regex needsFunctionConstantsRegex(R"("NeedsFunctionConstants"\s*:\s*(true|false))");
        if (std::regex_search(json, match, needsFunctionConstantsRegex))
            reflection.needsFunctionConstants = match[1].str() == "true";

        std::regex resourceCountRegex(R"("ResourceCount"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, resourceCountRegex))
        {
            reflection.resourceCount = static_cast<uint32_t>(std::stoul(match[1].str()));
            hasResourceCount = true;
        }

        // MSC emits one compact object per top-level argument-buffer entry. The
        // entry order is the physical descriptor-table index used by the runtime.
        std::regex entryRegex(
            R"msc(\{"EltOffset"\s*:\s*([0-9]+)\s*,\s*"Name"\s*:\s*"[^"]*"\s*,\s*"Size"\s*:\s*([0-9]+)\s*,\s*"Slot"\s*:\s*([0-9]+)\s*,\s*"Space"\s*:\s*([0-9]+)\s*,\s*"Type"\s*:\s*"([^"]+)"\})msc");
        for (std::sregex_iterator it(json.begin(), json.end(), entryRegex), end; it != end; ++it)
        {
            /*
             * *it[5] corresponds to 5th argument for a resource in top level argument buffer
             * Ex: a single resource can have this reflection: {"EltOffset":120,"Name":"","Size":24,"Slot":1,"Space":0,"Type":"CBV"}
             * regex capture groups:
             [0] whole matched object
             * [1] EltOffset  -> "120"
             * [2] Size       -> "24"
             * [3] Slot       -> "1"
             * [4] Space      -> "0"
             * [5] Type       -> "CBV"
             * the 5th arg here is the "Type". so default is to SRV, and then check for actual type, and change to the type post that
            */
            MscArgumentType type = MscArgumentType::SRV;
            if (!parseMscArgumentType((*it)[5].str(), type))
                continue;

            const uint32_t elementOffset = static_cast<uint32_t>(std::stoul((*it)[1].str()));
            const uint32_t elementSize = static_cast<uint32_t>(std::stoul((*it)[2].str()));

            MscArgumentBinding binding;
            binding.index = elementSize ? elementOffset / elementSize : static_cast<uint32_t>(reflection.topLevelArgumentBuffer.size());
            binding.slot = static_cast<uint32_t>(std::stoul((*it)[3].str()));
            binding.space = static_cast<uint32_t>(std::stoul((*it)[4].str()));
            binding.type = type;
            reflection.topLevelArgumentBuffer.push_back(binding);
        }

        std::regex vertexInputsRegex(R"("vertex_inputs"\s*:\s*\[([^\]]*)\])");
        if (std::regex_search(json, match, vertexInputsRegex))
        {
            const std::string vertexInputs = match[1].str();
            std::regex objectRegex(R"(\{[^\}]*\})");
            std::regex indexRegex(R"("index"\s*:\s*([0-9]+))");
            std::regex nameRegex(R"name("name"\s*:\s*"([^"]+)")name");
            for (std::sregex_iterator it(vertexInputs.begin(), vertexInputs.end(), objectRegex), end; it != end; ++it)
            {
                const std::string object = it->str();
                std::smatch indexMatch;
                std::smatch nameMatch;
                if (!std::regex_search(object, indexMatch, indexRegex) ||
                    !std::regex_search(object, nameMatch, nameRegex))
                    continue;

                const std::string name = normalizeMscVertexInputName(nameMatch[1].str());
                if (!name.empty())
                    reflection.vertexInputAttributes[name] =
                        static_cast<uint32_t>(std::stoul(indexMatch[1].str()));
            }
        }

        // size of one VS output record consumed by the emulated GS mesh stage.
        std::regex vertexOutputSizeRegex(R"("vertex_output_size_in_bytes"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, vertexOutputSizeRegex))
            reflection.vertexOutputSizeInBytes = static_cast<uint32_t>(std::stoul(match[1].str()));

        // max number of source primitives the generated mesh threadgroup processes.
        std::regex maxInputPrimsRegex(R"("max_input_primitives_per_mesh_threadgroup"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, maxInputPrimsRegex))
            reflection.maxInputPrimitivesPerMeshThreadgroup = static_cast<uint32_t>(std::stoul(match[1].str()));

        // GS instance count used by the IRConverter emulation config.
        std::regex instanceCountRegex(R"("instance_count"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, instanceCountRegex))
            reflection.geometryInstanceCount = static_cast<uint32_t>(std::stoul(match[1].str()));

        // source primitive topology expected by the converted geometry shader.
        std::regex inputPrimitiveRegex(R"name("input_primitive"\s*:\s*"([^"]+)")name");
        if (std::regex_search(json, match, inputPrimitiveRegex))
            reflection.inputPrimitive = match[1].str();
        
        std::regex payloadRegex(R"("max_payload_size_in_bytes"\s*:\s*([0-9]+))");
        if (std::regex_search(json, match, payloadRegex))
            reflection.maxPayloadSizeInBytes = static_cast<uint32_t>(std::stoul(match[1].str()));

        if (computeThreads)
        {
            std::regex tgSizeRegex(R"re("(?:tg_size|num_threads)"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\])re");
            if (std::regex_search(json, match, tgSizeRegex))
            {
                computeThreads->width = static_cast<NSUInteger>(std::stoul(match[1].str()));
                computeThreads->height = static_cast<NSUInteger>(std::stoul(match[2].str()));
                computeThreads->depth = static_cast<NSUInteger>(std::stoul(match[3].str()));
            }
        }

        if (reflection.resourceCount == 0)
            reflection.resourceCount = static_cast<uint32_t>(reflection.topLevelArgumentBuffer.size());

        // Shaders that consume only vertex/pixel inputs legitimately have an
        // empty top-level argument buffer. Treat explicit ResourceCount:0 as
        // valid reflection instead of falling back to legacy ordering.
        reflection.valid = !reflection.topLevelArgumentBuffer.empty() ||
            (hasResourceCount && reflection.resourceCount == 0);
        return reflection.valid;
    }

    static std::vector<std::filesystem::path> makeMscReflectionCandidates(const std::string& debugName)
    {
        std::vector<std::filesystem::path> candidates;
        if (debugName.empty())
            return candidates;

        std::filesystem::path path(debugName);
        if (path.has_extension())
        {
            std::filesystem::path reflectionPath = path;
            reflectionPath.replace_extension(".reflection.json");
            candidates.push_back(reflectionPath);

            reflectionPath = path;
            reflectionPath.replace_extension(".json");
            candidates.push_back(reflectionPath);
        }

        candidates.emplace_back(debugName + ".reflection.json");
        candidates.emplace_back(debugName + ".json");
        return candidates;
    }

    static bool loadMscReflection(const std::string& debugName, MscShaderReflection& reflection, MTLSize* computeThreads, std::string& loadedPath)
    {
        for (const std::filesystem::path& path : makeMscReflectionCandidates(debugName))
        {
            std::string json;
            if (!readTextFile(path.string(), json))
                continue;

            MscShaderReflection parsed;
            if (!parseMscReflectionJson(json, parsed, computeThreads))
                continue;

            reflection = std::move(parsed);
            loadedPath = path.string();
            return true;
        }

        return false;
    }

    static std::filesystem::path makeMscStageInLibraryPath(const std::string& debugName)
    {
        if (debugName.empty())
            return {};

        std::filesystem::path path(debugName);
        if (path.extension() == ".metallib")
        {
            std::string value = path.string();
            constexpr const char* suffix = ".metallib";
            value.resize(value.size() - std::strlen(suffix));
            value += ".stageIn.metallib";
            return value;
        }

        return {};
    }

    static std::string converterErrorDescription(const IRError* error)
    {
        if (!error)
            return "no converter diagnostic";
        const uint32_t code = IRErrorGetCode(error);
        std::string message = "MSC error " + std::to_string(code);
        switch (code)
        {
        case IRErrorCodeShaderRequiresRootSignature:
            return message + ": shader requires an explicit root signature with directly indexed heap flags";
        case IRErrorCodeUnrecognizedRootSignatureDescriptor:
            return message + ": converter rejected the root signature descriptor version or flags";
        case IRErrorCodeUnrecognizedParameterTypeInRootSignature:
            return message + ": converter rejected a root parameter type";
        case IRErrorCodeResourceNotReferencedByRootSignature:
            return message + ": a shader register, space, or descriptor array extent is absent from the root signature";
        case IRErrorCodeUnsupportedInstruction:
            return message + ": DXIL contains an instruction unsupported by this converter";
        case IRErrorCodeCompilationError:
            return message + ": converter compilation failed";
        case IRErrorCodeUnableToVerifyModule:
            return message + ": converter could not verify the DXIL module";
        case IRErrorCodeUnableToLinkModule:
            return message + ": converter could not link the module; check SDK and deployment target compatibility";
        case IRErrorCodeUnrecognizedDXILHeader:
            return message + ": input is not a supported DXIL container";
        default:
            return message;
        }
    }

    template<typename T>
    struct DxcRelease
    {
        void operator()(T* object) const
        {
            if (object)
                object->Release();
        }
    };

    static IRRootSignature* createDxilRootSignature(const MTL3Context& context, Shader& shader,
        const void* binary, size_t binarySize)
    {
        IDxcUtils* rawUtils = nullptr;
        HRESULT result = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&rawUtils));
        std::unique_ptr<IDxcUtils, DxcRelease<IDxcUtils>> utils(rawUtils);
        if (FAILED(result) || !utils)
        {
            context.error("[metal3] Cannot initialize DXC resource reflection for '" + shader.desc.debugName +
                "', HRESULT " + std::to_string(uint32_t(result)));
            return nullptr;
        }
        const DxcBuffer buffer{binary, binarySize, 0};
        ID3D12ShaderReflection* rawReflection = nullptr;
        result = utils->CreateReflection(&buffer, IID_PPV_ARGS(&rawReflection));
        std::unique_ptr<ID3D12ShaderReflection, DxcRelease<ID3D12ShaderReflection>> reflection(rawReflection);
        D3D12_SHADER_DESC desc{};
        if (FAILED(result) || !reflection || FAILED(reflection->GetDesc(&desc)))
        {
            context.error("[metal3] Cannot reflect DXIL resources for '" + shader.desc.debugName +
                "', HRESULT " + std::to_string(uint32_t(result)) + "; preserve DXIL reflection data");
            return nullptr;
        }

        std::vector<std::vector<IRDescriptorRange1>> ranges(2);
        std::vector<MscDescriptorTable> tables(2);
        tables[1].type = MscArgumentType::Sampler;
        auto& metadata = shader.mscReflection;
        for (uint32_t i = 0; i < desc.BoundResources; ++i)
        {
            D3D12_SHADER_INPUT_BIND_DESC resource{};
            if (FAILED(reflection->GetResourceBindingDesc(i, &resource)))
            {
                context.error("[metal3] Cannot reflect DXIL resource " + std::to_string(i) + " for '" + shader.desc.debugName + "'");
                return nullptr;
            }
            MscArgumentBinding binding;
            IRDescriptorRange1 range{};
            switch (resource.Type)
            {
            case D3D_SIT_CBUFFER:
                binding.type = MscArgumentType::CBV;
                range.RangeType = IRDescriptorRangeTypeCBV;
                break;
            case D3D_SIT_TEXTURE:
            case D3D_SIT_TBUFFER:
            case D3D_SIT_STRUCTURED:
            case D3D_SIT_BYTEADDRESS:
            case D3D_SIT_RTACCELERATIONSTRUCTURE:
                binding.type = MscArgumentType::SRV;
                range.RangeType = IRDescriptorRangeTypeSRV;
                break;
            case D3D_SIT_UAV_RWTYPED:
            case D3D_SIT_UAV_RWSTRUCTURED:
            case D3D_SIT_UAV_RWBYTEADDRESS:
                binding.type = MscArgumentType::UAV;
                range.RangeType = IRDescriptorRangeTypeUAV;
                break;
            case D3D_SIT_SAMPLER:
                binding.type = MscArgumentType::Sampler;
                range.RangeType = IRDescriptorRangeTypeSampler;
                break;
            default:
                context.error("[metal3] Unsupported DXIL resource type " + std::to_string(resource.Type) +
                    " for '" + shader.desc.debugName + "', resource '" + (resource.Name ? resource.Name : "") +
                    "', slot " + std::to_string(resource.BindPoint) + ", space " + std::to_string(resource.Space) +
                    "; UAV counters and ray tracing root resources are not supported");
                return nullptr;
            }
            const bool unbounded = resource.BindCount == 0 || resource.BindCount == UINT32_MAX;
            uint32_t tableIndex = binding.type == MscArgumentType::Sampler ? 1 : 0;
            if (unbounded)
            {
                tableIndex = uint32_t(tables.size());
                tables.emplace_back();
                tables.back().slot = resource.BindPoint;
                tables.back().space = resource.Space;
                tables.back().type = binding.type;
                ranges.emplace_back();
            }
            auto& table = tables[tableIndex];
            range.NumDescriptors = unbounded ? UINT32_MAX : resource.BindCount;
            range.BaseShaderRegister = resource.BindPoint;
            range.RegisterSpace = resource.Space;
            range.Flags = static_cast<IRDescriptorRangeFlags>(IRDescriptorRangeFlagDescriptorsVolatile |
                (binding.type == MscArgumentType::Sampler ? 0 : IRDescriptorRangeFlagDataVolatile));
            range.OffsetInDescriptorsFromTableStart = table.descriptorCount;
            ranges[tableIndex].push_back(range);
            if (!unbounded)
            {
                if (resource.BindCount > UINT32_MAX / sizeof(IRDescriptorTableEntry) - table.descriptorCount ||
                    resource.BindCount - 1 > UINT32_MAX - resource.BindPoint)
                {
                    context.error("[metal3] DXIL descriptor range exceeds the Metal argument buffer limit: " + shader.desc.debugName);
                    return nullptr;
                }
                binding.tableIndex = tableIndex;
                binding.index = table.descriptorCount;
                binding.byteOffset = table.descriptorCount * sizeof(IRDescriptorTableEntry);
                binding.sizeBytes = uint64_t(resource.BindCount) * sizeof(IRDescriptorTableEntry);
                binding.slot = resource.BindPoint;
                binding.space = resource.Space;
                metadata.topLevelArgumentBuffer.push_back(binding);
                table.descriptorCount += resource.BindCount;
            }
        }
        std::vector<IRRootParameter1> parameters;
        std::vector<uint32_t> tableIndices;
        for (uint32_t i = 0; i < tables.size(); ++i)
        {
            if (ranges[i].empty())
                continue;
            IRRootParameter1 parameter{};
            parameter.ParameterType = IRRootParameterTypeDescriptorTable;
            parameter.ShaderVisibility = IRShaderVisibilityAll;
            parameter.DescriptorTable.NumDescriptorRanges = uint32_t(ranges[i].size());
            parameter.DescriptorTable.pDescriptorRanges = ranges[i].data();
            parameters.push_back(parameter);
            tableIndices.push_back(i);
        }
        IRVersionedRootSignatureDescriptor rootDesc{};
        rootDesc.version = IRRootSignatureVersion_1_1;
        rootDesc.desc_1_1.NumParameters = uint32_t(parameters.size());
        rootDesc.desc_1_1.pParameters = parameters.data();
        uint32_t flags = IRRootSignatureFlagAllowInputAssemblerInputLayout;
        const uint64_t requiresFlags = reflection->GetRequiresFlags();
        metadata.directlyIndexedResourceHeap = (requiresFlags & D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING) != 0;
        metadata.directlyIndexedSamplerHeap = (requiresFlags & D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING) != 0;
        if (requiresFlags & D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING)
            flags |= IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed;
        if (requiresFlags & D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING)
            flags |= IRRootSignatureFlagSamplerHeapDirectlyIndexed;
        rootDesc.desc_1_1.Flags = static_cast<IRRootSignatureFlags>(flags);
        IRError* error = nullptr;
        std::unique_ptr<IRRootSignature, decltype(&IRRootSignatureDestroy)> root(
            IRRootSignatureCreateFromDescriptor(&rootDesc, &error), IRRootSignatureDestroy);
        if (!root)
            context.error("[metal3] Cannot create root signature for '" + shader.desc.debugName + "': " + converterErrorDescription(error));
        if (error)
            IRErrorDestroy(error);
        if (!root)
            return nullptr;
        std::vector<IRResourceLocation> locations(IRRootSignatureGetResourceCount(root.get()));
        IRRootSignatureGetResourceLocations(root.get(), locations.data());
        if (locations.size() != parameters.size())
        {
            context.error("[metal3] MSC root reflection does not match the descriptor table count: " + shader.desc.debugName);
            return nullptr;
        }
        for (size_t i = 0; i < locations.size(); ++i)
        {
            const auto& location = locations[i];
            if (location.resourceType != IRResourceTypeTable || location.sizeBytes != sizeof(uint64_t) ||
                location.topLevelOffset > UINT32_MAX - sizeof(uint64_t))
            {
                context.error("[metal3] Unsupported MSC descriptor table pointer layout: " + shader.desc.debugName);
                return nullptr;
            }
            tables[tableIndices[i]].byteOffset = location.topLevelOffset;
            metadata.argumentBufferSize = std::max(metadata.argumentBufferSize, location.topLevelOffset + location.sizeBytes);
        }
        for (auto& binding : metadata.topLevelArgumentBuffer)
            binding.tableIndex = uint32_t(std::find(tableIndices.begin(), tableIndices.end(), binding.tableIndex) - tableIndices.begin());
        for (uint32_t index : tableIndices)
            metadata.descriptorTables.push_back(tables[index]);
        metadata.resourceCount = desc.BoundResources;
        return root.release();
    }

    static bool convertDxilShader(const MTL3Context& context, Shader& shader, const void* binary, size_t binarySize,
        std::vector<uint8_t>& metallib, std::string& entryName)
    {
        IRShaderStage stage = IRShaderStageInvalid;
        switch (shader.desc.shaderType)
        {
        case ShaderType::Vertex: stage = IRShaderStageVertex; break;
        case ShaderType::Pixel: stage = IRShaderStageFragment; break;
        case ShaderType::Compute: stage = IRShaderStageCompute; break;
        case ShaderType::Mesh: stage = IRShaderStageMesh; break;
        case ShaderType::Amplification: stage = IRShaderStageAmplification; break;
        default:
            context.error("[metal3] DXIL conversion supports vertex, pixel, compute, mesh and amplification shaders: " + shader.desc.debugName);
            return false;
        }
        std::unique_ptr<IRRootSignature, decltype(&IRRootSignatureDestroy)> root(
            createDxilRootSignature(context, shader, binary, binarySize), IRRootSignatureDestroy);
        if (!root)
            return false;

        std::unique_ptr<IRCompiler, decltype(&IRCompilerDestroy)> compiler(IRCompilerCreate(), IRCompilerDestroy);
        std::unique_ptr<IRObject, decltype(&IRObjectDestroy)> input(
            IRObjectCreateFromDXIL(static_cast<const uint8_t*>(binary), binarySize, IRBytecodeOwnershipNone), IRObjectDestroy);
        if (!compiler || !input)
        {
            context.error("[metal3] Unable to initialize DXIL conversion: " + shader.desc.debugName);
            return false;
        }

        IRCompilerSetMinimumGPUFamily(compiler.get(), IRGPUFamilyMetal3);
        IRCompilerSetMinimumDeploymentTarget(compiler.get(), IROperatingSystem_macOS, "14.0.0");
        IRCompilerSetStageInGenerationMode(compiler.get(), IRStageInCodeGenerationModeUseMetalVertexFetch);
        IRCompilerSetFunctionConstantResourceSpace(compiler.get(), UINT32_MAX);
        IRCompilerSetFramebufferFetchResourceSpace(compiler.get(), UINT32_MAX);
        IRCompilerIgnoreRootSignature(compiler.get(), true);
        IRCompilerSetGlobalRootSignature(compiler.get(), root.get());
        IRCompilerSetEntryPointName(compiler.get(), entryName.c_str());
        IRError* error = nullptr;
        std::unique_ptr<IRObject, decltype(&IRObjectDestroy)> output(
            IRCompilerAllocCompileAndLink(compiler.get(), entryName.c_str(), input.get(), &error), IRObjectDestroy);
        if (!output)
        {
            context.error("[metal3] DXIL conversion failed for '" + shader.desc.debugName + "': " + converterErrorDescription(error));
            if (error)
                IRErrorDestroy(error);
            return false;
        }
        if (error)
            IRErrorDestroy(error);

        std::unique_ptr<IRShaderReflection, decltype(&IRShaderReflectionDestroy)> reflection(
            IRShaderReflectionCreate(), IRShaderReflectionDestroy);
        std::unique_ptr<IRMetalLibBinary, decltype(&IRMetalLibBinaryDestroy)> library(
            IRMetalLibBinaryCreate(), IRMetalLibBinaryDestroy);
        if (!reflection || !library || !IRObjectGetReflection(output.get(), stage, reflection.get()) ||
            !IRObjectGetMetalLibBinary(output.get(), stage, library.get()))
        {
            context.error("[metal3] Missing MSC library or stage reflection: " + shader.desc.debugName);
            return false;
        }

        auto& metadata = shader.mscReflection;
        metadata.needsFunctionConstants = IRShaderReflectionNeedsFunctionConstants(reflection.get());
        if (metadata.needsFunctionConstants)
        {
            context.error("[metal3] Shader requires unsupported MSC function constants: " + shader.desc.debugName);
            return false;
        }
        const char* reflectedEntry = IRShaderReflectionGetEntryPointFunctionName(reflection.get());
        if (!reflectedEntry || !*reflectedEntry)
        {
            context.error("[metal3] Missing MSC entry point: " + shader.desc.debugName);
            return false;
        }
        entryName = reflectedEntry;

        if (stage == IRShaderStageVertex)
        {
            IRVersionedVSInfo info{};
            if (!IRShaderReflectionCopyVertexInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC vertex inputs: " + shader.desc.debugName);
                return false;
            }
            metadata.vertexOutputSizeInBytes = info.info_1_0.vertex_output_size_in_bytes;
            for (size_t i = 0; i < info.info_1_0.num_vertex_inputs; ++i)
            {
                const auto& attribute = info.info_1_0.vertex_inputs[i];
                if (attribute.name)
                    metadata.vertexInputAttributes[normalizeMscVertexInputName(attribute.name)] = attribute.attributeIndex;
            }
            IRShaderReflectionReleaseVertexInfo(&info);
        }
        if (stage == IRShaderStageCompute)
        {
            IRVersionedCSInfo info{};
            if (!IRShaderReflectionCopyComputeInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC compute dimensions: " + shader.desc.debugName);
                return false;
            }
            shader.computeThreadsPerGroup = MTLSizeMake(info.info_1_0.tg_size[0], info.info_1_0.tg_size[1], info.info_1_0.tg_size[2]);
            IRShaderReflectionReleaseComputeInfo(&info);
            shader.computeThreadsPerGroupValid = shader.computeThreadsPerGroup.width != 0 &&
                shader.computeThreadsPerGroup.height != 0 && shader.computeThreadsPerGroup.depth != 0;
            if (!shader.computeThreadsPerGroupValid)
            {
                context.error("[metal3] Invalid MSC compute dimensions: " + shader.desc.debugName);
                return false;
            }
        }
        if (stage == IRShaderStageMesh)
        {
            IRVersionedMSInfo info{};
            if (!IRShaderReflectionCopyMeshInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC mesh dimensions: " + shader.desc.debugName);
                return false;
            }
            shader.computeThreadsPerGroup = MTLSizeMake(info.info_1_0.num_threads[0], info.info_1_0.num_threads[1], info.info_1_0.num_threads[2]);
            metadata.maxPayloadSizeInBytes = info.info_1_0.max_payload_size_in_bytes;
            IRShaderReflectionReleaseMeshInfo(&info);
            shader.computeThreadsPerGroupValid = shader.computeThreadsPerGroup.width != 0 &&
                shader.computeThreadsPerGroup.height != 0 && shader.computeThreadsPerGroup.depth != 0;
        }
        if (stage == IRShaderStageAmplification)
        {
            IRVersionedASInfo info{};
            if (!IRShaderReflectionCopyAmplificationInfo(reflection.get(), IRReflectionVersion_1_0, &info))
            {
                context.error("[metal3] Missing MSC amplification dimensions: " + shader.desc.debugName);
                return false;
            }
            shader.computeThreadsPerGroup = MTLSizeMake(info.info_1_0.num_threads[0], info.info_1_0.num_threads[1], info.info_1_0.num_threads[2]);
            metadata.maxPayloadSizeInBytes = info.info_1_0.max_payload_size_in_bytes;
            IRShaderReflectionReleaseAmplificationInfo(&info);
            shader.computeThreadsPerGroupValid = shader.computeThreadsPerGroup.width != 0 &&
                shader.computeThreadsPerGroup.height != 0 && shader.computeThreadsPerGroup.depth != 0;
        }
        metallib.resize(IRMetalLibGetBytecodeSize(library.get()));
        if (metallib.empty() || IRMetalLibGetBytecode(library.get(), metallib.data()) != metallib.size())
        {
            context.error("[metal3] Failed to extract MSC bytecode: " + shader.desc.debugName);
            return false;
        }
        metadata.valid = true;
        return true;
    }

    constexpr uint32_t c_ShaderLibraryCacheFormatVersion = 1;
    constexpr uint32_t c_ShaderLibraryCacheMagic = 0x334c544d;
    constexpr char c_ShaderLibraryCacheSettings[] =
        "metal-dxil;column-major;linear;vertex-fetch-stage-in;gpu-family-metal3;macos-14.0.0;ignore-root-signature;"
        "global-root-signature;function-constants-none;framebuffer-fetch-none;sm6.6";

    struct ShaderLibraryCacheHeader
    {
        uint32_t magic;
        uint32_t version;
        uint32_t stage;
        uint32_t reserved;
        uint64_t key;
        uint64_t secondaryKey;
        uint64_t dxilSize;
        uint64_t metallibSize;
        uint64_t metadataSize;
    };

    struct ShaderLibraryCacheMetadata
    {
        MscShaderReflection reflection;
        MTLSize computeThreadsPerGroup = MTLSizeMake(1, 1, 1);
        bool computeThreadsPerGroupValid = false;
        std::string reflectedEntryName;
    };

    struct ShaderLibraryCacheWriter
    {
        std::vector<uint8_t> data;

        void writeBytes(const void* bytes, size_t size)
        {
            const uint8_t* begin = static_cast<const uint8_t*>(bytes);
            data.insert(data.end(), begin, begin + size);
        }
        void writeU8(uint8_t value) { data.push_back(value); }
        void writeU32(uint32_t value) { writeBytes(&value, sizeof(value)); }
        void writeU64(uint64_t value) { writeBytes(&value, sizeof(value)); }
        void writeString(const std::string& value)
        {
            writeU32(uint32_t(value.size()));
            writeBytes(value.data(), value.size());
        }
    };

    struct ShaderLibraryCacheReader
    {
        const uint8_t* data;
        size_t size;
        size_t offset = 0;
        bool ok = true;

        ShaderLibraryCacheReader(const uint8_t* bytes, size_t byteCount)
            : data(bytes), size(byteCount)
        {
        }

        bool readBytes(void* out, size_t count)
        {
            if (!ok || count > size - offset)
            {
                ok = false;
                return false;
            }
            std::memcpy(out, data + offset, count);
            offset += count;
            return true;
        }

        uint8_t readU8()
        {
            uint8_t value = 0;
            readBytes(&value, sizeof(value));
            return value;
        }

        uint32_t readU32()
        {
            uint32_t value = 0;
            readBytes(&value, sizeof(value));
            return value;
        }

        uint64_t readU64()
        {
            uint64_t value = 0;
            readBytes(&value, sizeof(value));
            return value;
        }

        bool readString(std::string& out)
        {
            const uint32_t length = readU32();
            if (!ok || length > size - offset)
            {
                ok = false;
                return false;
            }
            out.assign(reinterpret_cast<const char*>(data + offset), length);
            offset += length;
            return true;
        }

        size_t remaining() const { return ok ? size - offset : 0; }
    };

    static uint64_t hashFnv1a(const void* bytes, size_t size, uint64_t hash)
    {
        const uint8_t* values = static_cast<const uint8_t*>(bytes);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= values[i];
            hash *= 0x100000001b3ull;
        }
        return hash;
    }

    static uint64_t hashScramble(const void* bytes, size_t size, uint64_t hash)
    {
        const uint8_t* values = static_cast<const uint8_t*>(bytes);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= uint64_t(values[i]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            hash = (hash << 31) | (hash >> 33);
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 29;
        }
        return hash;
    }

    static void computeShaderLibraryCacheKeys(const ShaderDesc& desc, const std::string& requestedEntryName,
        const void* dxil, size_t dxilSize, uint64_t& outKey, uint64_t& outSecondaryKey)
    {
        const uint32_t formatVersion = c_ShaderLibraryCacheFormatVersion;
        const uint32_t toolchainVersion[3] = { IR_VERSION_MAJOR, IR_VERSION_MINOR, IR_VERSION_PATCH };
        const uint32_t stage = uint32_t(desc.shaderType);
        const uint64_t sourceSize = uint64_t(dxilSize);

        uint64_t key = hashFnv1a(&formatVersion, sizeof(formatVersion), 0xcbf29ce484222325ull);
        key = hashFnv1a(toolchainVersion, sizeof(toolchainVersion), key);
        key = hashFnv1a(c_ShaderLibraryCacheSettings, sizeof(c_ShaderLibraryCacheSettings) - 1, key);
        key = hashFnv1a(&stage, sizeof(stage), key);
        key = hashFnv1a(requestedEntryName.data(), requestedEntryName.size(), key);
        key = hashFnv1a(&sourceSize, sizeof(sourceSize), key);
        key = hashFnv1a(dxil, dxilSize, key);

        uint64_t secondaryKey = hashScramble(&formatVersion, sizeof(formatVersion), 0x8ebc6af09c88c6e3ull);
        secondaryKey = hashScramble(toolchainVersion, sizeof(toolchainVersion), secondaryKey);
        secondaryKey = hashScramble(c_ShaderLibraryCacheSettings, sizeof(c_ShaderLibraryCacheSettings) - 1, secondaryKey);
        secondaryKey = hashScramble(&stage, sizeof(stage), secondaryKey);
        secondaryKey = hashScramble(requestedEntryName.data(), requestedEntryName.size(), secondaryKey);
        secondaryKey = hashScramble(&sourceSize, sizeof(sourceSize), secondaryKey);
        secondaryKey = hashScramble(dxil, dxilSize, secondaryKey);

        outKey = key;
        outSecondaryKey = secondaryKey;
    }

    static std::filesystem::path makeShaderLibraryCachePath(const std::string& directory, uint64_t key)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "%016llx.mtl3lib", static_cast<unsigned long long>(key));
        return std::filesystem::path(directory) / name;
    }

    static void serializeShaderLibraryMetadata(const Shader& shader, const std::string& reflectedEntryName,
        std::vector<uint8_t>& outMetadata)
    {
        ShaderLibraryCacheWriter writer;
        writer.writeString(reflectedEntryName);
        writer.writeU64(uint64_t(shader.computeThreadsPerGroup.width));
        writer.writeU64(uint64_t(shader.computeThreadsPerGroup.height));
        writer.writeU64(uint64_t(shader.computeThreadsPerGroup.depth));
        writer.writeU8(shader.computeThreadsPerGroupValid ? 1 : 0);

        const MscShaderReflection& reflection = shader.mscReflection;
        writer.writeU8(reflection.valid ? 1 : 0);
        writer.writeU8(reflection.needsFunctionConstants ? 1 : 0);
        writer.writeU8(reflection.directlyIndexedResourceHeap ? 1 : 0);
        writer.writeU8(reflection.directlyIndexedSamplerHeap ? 1 : 0);
        writer.writeU32(reflection.resourceCount);
        writer.writeU64(reflection.argumentBufferSize);
        writer.writeU32(reflection.vertexOutputSizeInBytes);
        writer.writeU32(reflection.maxInputPrimitivesPerMeshThreadgroup);
        writer.writeU32(reflection.geometryInstanceCount);
        writer.writeU32(reflection.maxPayloadSizeInBytes);
        writer.writeString(reflection.shaderType);
        writer.writeString(reflection.inputPrimitive);

        writer.writeU32(uint32_t(reflection.topLevelArgumentBuffer.size()));
        for (const MscArgumentBinding& binding : reflection.topLevelArgumentBuffer)
        {
            writer.writeU32(binding.index);
            writer.writeU32(binding.slot);
            writer.writeU32(binding.space);
            writer.writeU8(uint8_t(binding.type));
            writer.writeU32(binding.byteOffset);
            writer.writeU64(binding.sizeBytes);
            writer.writeU32(binding.tableIndex);
        }

        writer.writeU32(uint32_t(reflection.descriptorTables.size()));
        for (const MscDescriptorTable& table : reflection.descriptorTables)
        {
            writer.writeU32(table.byteOffset);
            writer.writeU32(table.descriptorCount);
            writer.writeU32(table.slot);
            writer.writeU32(table.space);
            writer.writeU8(uint8_t(table.type));
            writer.writeU32(table.layoutIndex);
            writer.writeU32(table.descriptorOffset);
        }

        writer.writeU32(uint32_t(reflection.vertexInputAttributes.size()));
        for (const auto& attribute : reflection.vertexInputAttributes)
        {
            writer.writeString(attribute.first);
            writer.writeU32(attribute.second);
        }

        outMetadata = std::move(writer.data);
    }

    static bool deserializeShaderLibraryMetadata(ShaderLibraryCacheReader& reader, ShaderLibraryCacheMetadata& metadata)
    {
        if (!reader.readString(metadata.reflectedEntryName) || metadata.reflectedEntryName.empty())
            return false;

        const uint64_t width = reader.readU64();
        const uint64_t height = reader.readU64();
        const uint64_t depth = reader.readU64();
        if (!reader.ok)
            return false;
        metadata.computeThreadsPerGroup = MTLSizeMake(NSUInteger(width), NSUInteger(height), NSUInteger(depth));
        metadata.computeThreadsPerGroupValid = reader.readU8() != 0;

        MscShaderReflection& reflection = metadata.reflection;
        reflection.valid = reader.readU8() != 0;
        reflection.needsFunctionConstants = reader.readU8() != 0;
        reflection.directlyIndexedResourceHeap = reader.readU8() != 0;
        reflection.directlyIndexedSamplerHeap = reader.readU8() != 0;
        reflection.resourceCount = reader.readU32();
        reflection.argumentBufferSize = reader.readU64();
        reflection.vertexOutputSizeInBytes = reader.readU32();
        reflection.maxInputPrimitivesPerMeshThreadgroup = reader.readU32();
        reflection.geometryInstanceCount = reader.readU32();
        reflection.maxPayloadSizeInBytes = reader.readU32();
        if (!reader.readString(reflection.shaderType) || !reader.readString(reflection.inputPrimitive))
            return false;

        const uint32_t bindingCount = reader.readU32();
        if (!reader.ok || bindingCount > reader.remaining() / 29)
            return false;
        reflection.topLevelArgumentBuffer.resize(bindingCount);
        for (MscArgumentBinding& binding : reflection.topLevelArgumentBuffer)
        {
            binding.index = reader.readU32();
            binding.slot = reader.readU32();
            binding.space = reader.readU32();
            const uint8_t type = reader.readU8();
            if (type > uint8_t(MscArgumentType::Sampler))
                return false;
            binding.type = MscArgumentType(type);
            binding.byteOffset = reader.readU32();
            binding.sizeBytes = reader.readU64();
            binding.tableIndex = reader.readU32();
        }

        const uint32_t tableCount = reader.readU32();
        if (!reader.ok || tableCount > reader.remaining() / 25)
            return false;
        reflection.descriptorTables.resize(tableCount);
        for (MscDescriptorTable& table : reflection.descriptorTables)
        {
            table.byteOffset = reader.readU32();
            table.descriptorCount = reader.readU32();
            table.slot = reader.readU32();
            table.space = reader.readU32();
            const uint8_t type = reader.readU8();
            if (type > uint8_t(MscArgumentType::Sampler))
                return false;
            table.type = MscArgumentType(type);
            table.layoutIndex = reader.readU32();
            table.descriptorOffset = reader.readU32();
        }

        const uint32_t attributeCount = reader.readU32();
        if (!reader.ok || attributeCount > reader.remaining() / 5)
            return false;
        for (uint32_t index = 0; index < attributeCount; ++index)
        {
            std::string name;
            if (!reader.readString(name))
                return false;
            const uint32_t attributeIndex = reader.readU32();
            reflection.vertexInputAttributes[name] = attributeIndex;
        }

        return reader.ok;
    }

    static bool readShaderLibraryCacheFile(const std::filesystem::path& path, std::vector<uint8_t>& outData)
    {
        std::error_code errorCode;
        const auto fileSize = std::filesystem::file_size(path, errorCode);
        if (errorCode || fileSize < sizeof(ShaderLibraryCacheHeader) || fileSize > (256ull << 20))
            return false;

        outData.resize(size_t(fileSize));
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            outData.clear();
            return false;
        }
        file.read(reinterpret_cast<char*>(outData.data()), std::streamsize(outData.size()));
        if (!file)
        {
            outData.clear();
            return false;
        }
        return true;
    }

    static bool loadShaderLibraryFromCache(const std::string& directory, const ShaderDesc& desc,
        const std::string& requestedEntryName, const void* dxil, size_t dxilSize, Shader& shader,
        std::string& reflectedEntryName, std::vector<uint8_t>& metallib)
    {
        if (directory.empty() || !dxil || dxilSize == 0)
            return false;

        uint64_t key = 0;
        uint64_t secondaryKey = 0;
        computeShaderLibraryCacheKeys(desc, requestedEntryName, dxil, dxilSize, key, secondaryKey);

        const std::filesystem::path path = makeShaderLibraryCachePath(directory, key);
        std::vector<uint8_t> fileData;
        if (!readShaderLibraryCacheFile(path, fileData))
            return false;

        ShaderLibraryCacheHeader header{};
        std::memcpy(&header, fileData.data(), sizeof(header));
        if (header.magic != c_ShaderLibraryCacheMagic || header.version != c_ShaderLibraryCacheFormatVersion ||
            header.key != key || header.secondaryKey != secondaryKey || header.stage != uint32_t(desc.shaderType) ||
            header.dxilSize != uint64_t(dxilSize))
            return false;

        const size_t payloadSize = fileData.size() - sizeof(header);
        if (header.metadataSize == 0 || header.metallibSize == 0 ||
            header.metadataSize > payloadSize || header.metallibSize > payloadSize - header.metadataSize)
            return false;

        ShaderLibraryCacheReader reader(fileData.data() + sizeof(header), size_t(header.metadataSize));
        ShaderLibraryCacheMetadata metadata;
        if (!deserializeShaderLibraryMetadata(reader, metadata) || !reader.ok || reader.remaining() != 0)
            return false;
        if (!metadata.reflection.valid)
            return false;

        const uint8_t* metallibBegin = fileData.data() + sizeof(header) + header.metadataSize;
        if (header.metallibSize < 8 || std::memcmp(metallibBegin, "MTLB", 4) != 0)
            return false;
        metallib.assign(metallibBegin, metallibBegin + header.metallibSize);

        shader.mscReflection = std::move(metadata.reflection);
        shader.computeThreadsPerGroup = metadata.computeThreadsPerGroup;
        shader.computeThreadsPerGroupValid = metadata.computeThreadsPerGroupValid;
        reflectedEntryName = metadata.reflectedEntryName;
        return true;
    }

    static void removeShaderLibraryFromCache(const std::string& directory, const ShaderDesc& desc,
        const std::string& requestedEntryName, const void* dxil, size_t dxilSize)
    {
        if (directory.empty() || !dxil || dxilSize == 0)
            return;

        uint64_t key = 0;
        uint64_t secondaryKey = 0;
        computeShaderLibraryCacheKeys(desc, requestedEntryName, dxil, dxilSize, key, secondaryKey);

        std::error_code errorCode;
        std::filesystem::remove(makeShaderLibraryCachePath(directory, key), errorCode);
    }

    static void storeShaderLibraryInCache(const MTL3Context& context, const std::string& directory, const ShaderDesc& desc,
        const std::string& requestedEntryName, const std::string& reflectedEntryName, const Shader& shader,
        const void* dxil, size_t dxilSize, const std::vector<uint8_t>& metallib)
    {
        if (directory.empty() || !dxil || dxilSize == 0 || metallib.empty())
            return;

        std::vector<uint8_t> metadata;
        serializeShaderLibraryMetadata(shader, reflectedEntryName, metadata);
        if (metadata.empty())
            return;

        uint64_t key = 0;
        uint64_t secondaryKey = 0;
        computeShaderLibraryCacheKeys(desc, requestedEntryName, dxil, dxilSize, key, secondaryKey);

        ShaderLibraryCacheHeader header{};
        header.magic = c_ShaderLibraryCacheMagic;
        header.version = c_ShaderLibraryCacheFormatVersion;
        header.stage = uint32_t(desc.shaderType);
        header.key = key;
        header.secondaryKey = secondaryKey;
        header.dxilSize = uint64_t(dxilSize);
        header.metallibSize = uint64_t(metallib.size());
        header.metadataSize = uint64_t(metadata.size());

        std::error_code errorCode;
        const std::filesystem::path directoryPath(directory);
        std::filesystem::create_directories(directoryPath, errorCode);

        const std::filesystem::path path = makeShaderLibraryCachePath(directory, key);
        std::filesystem::path temporaryPath = path;
        temporaryPath += ".tmp";

        {
            std::ofstream file(temporaryPath, std::ios::binary | std::ios::trunc);
            if (!file)
                return;
            file.write(reinterpret_cast<const char*>(&header), static_cast<std::streamsize>(sizeof(header)));
            file.write(reinterpret_cast<const char*>(metadata.data()), std::streamsize(metadata.size()));
            file.write(reinterpret_cast<const char*>(metallib.data()), std::streamsize(metallib.size()));
            file.flush();
            if (!file)
            {
                std::error_code removeError;
                std::filesystem::remove(temporaryPath, removeError);
                return;
            }
        }

        std::filesystem::rename(temporaryPath, path, errorCode);
        if (errorCode)
        {
            context.warning("[metal3] failed to store the shader library cache entry '" + path.string() + "'");
            std::error_code removeError;
            std::filesystem::remove(temporaryPath, removeError);
        }
    }

    ShaderHandle Device::createShader(const ShaderDesc& d, const void* binary, size_t binarySize)
    {
        if (!binary || binarySize == 0)
            return nullptr;

        Shader* shader = new Shader();
        shader->desc = d;
        shader->bytecode.assign(static_cast<const uint8_t*>(binary), static_cast<const uint8_t*>(binary) + binarySize);
        const bool isDxil = binarySize >= 4 && std::memcmp(binary, "DXBC", 4) == 0;
        const void* source = binary;
        const size_t sourceSize = binarySize;
        const std::string requestedEntryName = d.entryName.empty() ? "main" : d.entryName;
        std::string entryName = requestedEntryName;
        std::vector<uint8_t> metallib;
        bool allowCache = true;
        bool cacheUsed = false;

        while (true)
        {
            const void* libraryData = source;
            size_t libraryDataSize = sourceSize;
            if (isDxil)
            {
                std::string reflectedEntryName = entryName;
                if (allowCache && loadShaderLibraryFromCache(m_ShaderLibraryCacheDirectory, d, requestedEntryName, source,
                    sourceSize, *shader, reflectedEntryName, metallib))
                {
                    entryName = reflectedEntryName;
                    cacheUsed = true;
                }
                else
                {
                    entryName = requestedEntryName;
                    shader->mscReflection = MscShaderReflection();
                    shader->computeThreadsPerGroup = MTLSizeMake(1, 1, 1);
                    shader->computeThreadsPerGroupValid = false;
                    if (!convertDxilShader(m_Context, *shader, source, sourceSize, metallib, entryName))
                    {
                        delete shader;
                        return nullptr;
                    }
                    cacheUsed = false;
                    storeShaderLibraryInCache(m_Context, m_ShaderLibraryCacheDirectory, d, requestedEntryName, entryName,
                        *shader, source, sourceSize, metallib);
                }
                libraryData = metallib.data();
                libraryDataSize = metallib.size();
            }

            dispatch_data_t data = dispatch_data_create(libraryData, libraryDataSize, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError* error = nil;
            shader->library = [m_Context.device newLibraryWithData:data error:&error];
            if (!shader->library)
            {
                std::string message = "[nvrhi] Failed to load Metal library";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                if (cacheUsed)
                {
                    m_Context.warning(message + "; discarding the cached shader library for '" + d.debugName + "'");
                    allowCache = false;
                    cacheUsed = false;
                    removeShaderLibraryFromCache(m_ShaderLibraryCacheDirectory, d, requestedEntryName, source, sourceSize);
                    continue;
                }
                m_Context.error(message);
                delete shader;
                return nullptr;
            }

            NSString* entry = [NSString stringWithUTF8String:entryName.c_str()];
            shader->function = [shader->library newFunctionWithName:entry];
            if (!shader->function)
            {
                NSString* objectEntry = [NSString stringWithFormat:@"%@.dxil_irconverter_object_shader", entry];
                shader->function = [shader->library newFunctionWithName:objectEntry];
            }
            if (shader->function)
                break;

            shader->library = nil;
            if (cacheUsed)
            {
                m_Context.warning("[metal3] cached shader library for '" + d.debugName +
                    "' has no entry point; discarding it");
                allowCache = false;
                cacheUsed = false;
                removeShaderLibraryFromCache(m_ShaderLibraryCacheDirectory, d, requestedEntryName, source, sourceSize);
                continue;
            }
            m_Context.error("[nvrhi] Failed to find Metal shader entry: " + d.entryName);
            delete shader;
            return nullptr;
        }

        std::filesystem::path stageInPath = isDxil ? std::filesystem::path() : makeMscStageInLibraryPath(d.debugName);
        if (!stageInPath.empty() && std::filesystem::exists(stageInPath))
        {
            NSError* stageInError = nil;
            shader->stageInLibrary = [m_Context.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:stageInPath.string().c_str()]]
                                                                    error:&stageInError];
            if (!shader->stageInLibrary)
            {
                std::string message = "[metal3] failed to load MSC stage-in library '" + stageInPath.string() + "'";
                if (stageInError)
                    message += std::string(": ") + [[stageInError localizedDescription] UTF8String];
                m_Context.warning(message);
            }
        }

        if (!d.debugName.empty())
            shader->function.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        MTLSize reflectionThreads = MTLSizeMake(0, 0, 0);
        std::string reflectionPath;
        if (isDxil)
        {
            reflectionThreads = shader->computeThreadsPerGroupValid ? shader->computeThreadsPerGroup : reflectionThreads;
        }
        else if (loadMscReflection(d.debugName, shader->mscReflection, &reflectionThreads, reflectionPath))
        {
            // uncomment for debugging
            // m_Context.info("[metal3] shader '" + d.debugName + "' reflection='" + reflectionPath +
            //     "' resources=" + std::to_string(shader->mscReflection.resourceCount) +
            //     " args=" + std::to_string(shader->mscReflection.topLevelArgumentBuffer.size()));
        }
        else
        {
            m_Context.warning("[metal3] shader '" + d.debugName + "' has no MSC reflection; using legacy argument-buffer ordering");
        }
        shader->reflectedBindingPlan = createMetalStageBindingPlan(d.shaderType, shader->mscReflection);

        if (d.shaderType == ShaderType::Compute || d.shaderType == ShaderType::Mesh || d.shaderType == ShaderType::Amplification)
        {
            if (reflectionThreads.width != 0 && reflectionThreads.height != 0 && reflectionThreads.depth != 0)
            {
                shader->computeThreadsPerGroup = reflectionThreads;
                shader->computeThreadsPerGroupValid = true;
            }

            if (shader->computeThreadsPerGroupValid)
            {
                // uncomment for debugging
                // m_Context.info("[metal3] compute shader '" + d.debugName + "' numthreads=" +
                //     std::to_string(shader->computeThreadsPerGroup.width) + "x" +
                //     std::to_string(shader->computeThreadsPerGroup.height) + "x" +
                //     std::to_string(shader->computeThreadsPerGroup.depth));
            }
        }

        return ShaderHandle::Create(shader);
    }

    // not really used anywhere, currently implemented cuz override
    // TODO: useful for creating a single shader bundle metallib file(s), path can (?) be added later
    ShaderLibraryHandle Device::createShaderLibrary(const void* binary, size_t binarySize)
    {
        if (!binary || binarySize == 0)
            return nullptr;

        ShaderLibrary* library = new ShaderLibrary();
        library->bytecode.assign(static_cast<const uint8_t*>(binary), static_cast<const uint8_t*>(binary) + binarySize);

        dispatch_data_t data = dispatch_data_create(binary, binarySize, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* error = nil;
        library->library = [m_Context.device newLibraryWithData:data error:&error];
        if (!library->library)
        {
            delete library;
            return nullptr;
        }

        return ShaderLibraryHandle::Create(library);
    }

    void Shader::getBytecode(const void** ppBytecode, size_t* pSize) const
    {
        if (ppBytecode) *ppBytecode = bytecode.data();
        if (pSize) *pSize = bytecode.size();
    }

    void ShaderLibrary::getBytecode(const void** ppBytecode, size_t* pSize) const
    {
        if (ppBytecode) *ppBytecode = bytecode.data();
        if (pSize) *pSize = bytecode.size();
    }

    ShaderHandle ShaderLibrary::getShader(const char* entryName, ShaderType shaderType)
    {
        Shader* shader = new Shader();
        shader->desc.entryName = entryName ? entryName : "main";
        shader->desc.shaderType = shaderType;
        shader->bytecode = bytecode;
        shader->library = library;
        shader->function = [library newFunctionWithName:[NSString stringWithUTF8String:shader->desc.entryName.c_str()]];
        if (!shader->function)
        {
            delete shader;
            return nullptr;
        }
        return ShaderHandle::Create(shader);
    }
}
