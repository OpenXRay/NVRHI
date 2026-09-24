#include "metal3-backend.h"
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#include <algorithm>
#include <cctype>

namespace nvrhi::metal3
{
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

    // c_MscVertexBufferBindPoint -> MSC's vertex-buffer binding slot where we start binding actual app vertex buffers
    // so vertex buffer slot 0 gets bound at index 6
    static constexpr uint32_t c_MscVertexBufferBindPoint = 6;
    // c_MscVertexAttributeBase -> the Metal vertex attribute index where MSC expects user vertex attributes to begin
    static constexpr uint32_t c_MscVertexAttributeBase = 11;
    // c_MetalMaxVertexAttributes -> just a guard for Metal’s vertex attribute descriptor array
    static constexpr uint32_t c_MetalMaxVertexAttributes = 31;


    InputLayoutHandle Device::createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader)
    {
        auto* shader = static_cast<Shader*>(vertexShader);
        InputLayout* layout = new InputLayout();
        layout->attributes.assign(d, d + attributeCount);
        layout->vertexDescriptor = [[MTLVertexDescriptor alloc] init];

        uint32_t location = 0;
        for (uint32_t index = 0; index < attributeCount; ++index)
        {
            const VertexAttributeDesc& attr = d[index];
            const uint32_t bufferIndex = c_MscVertexBufferBindPoint + attr.bufferIndex;
            const MTLVertexFormat format = convertVertexFormat(attr.format);
            if (!attr.arraySize || bufferIndex >= 31 || format == MTLVertexFormatInvalid)
            {
                m_Context.error("[metal3] Invalid vertex attribute format, array size or buffer index: " + attr.name);
                delete layout;
                return nullptr;
            }
            const std::string semantic = normalizeMscVertexInputName(attr.name);
            const size_t suffix = semantic.find_last_not_of("0123456789");
            const std::string prefix = semantic.substr(0, suffix + 1);
            const uint32_t firstSemantic = static_cast<uint32_t>(std::stoul(semantic.substr(suffix + 1)));
            for (uint32_t element = 0; element < attr.arraySize; ++element, ++location)
            {
                uint32_t attributeIndex = c_MscVertexAttributeBase + location;
                if (shader && !shader->mscReflection.vertexInputAttributes.empty())
                {
                    const auto found = shader->mscReflection.vertexInputAttributes.find(prefix + std::to_string(firstSemantic + element));
                    if (found == shader->mscReflection.vertexInputAttributes.end())
                        continue;
                    attributeIndex = c_MscVertexAttributeBase + found->second;
                }
                if (attributeIndex >= c_MetalMaxVertexAttributes)
                {
                    m_Context.error("[metal3] Vertex attribute exceeds Metal attribute limits: " + attr.name);
                    delete layout;
                    return nullptr;
                }
                layout->vertexDescriptor.attributes[attributeIndex].format = format;
                layout->vertexDescriptor.attributes[attributeIndex].offset = attr.offset + element * getFormatInfo(attr.format).bytesPerBlock;
                layout->vertexDescriptor.attributes[attributeIndex].bufferIndex = bufferIndex;
            }
            layout->vertexDescriptor.layouts[bufferIndex].stride = attr.elementStride;
            layout->vertexDescriptor.layouts[bufferIndex].stepFunction = attr.isInstanced ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            layout->vertexDescriptor.layouts[bufferIndex].stepRate = 1;
        }

        return InputLayoutHandle::Create(layout);
    }

    const VertexAttributeDesc* InputLayout::getAttributeDesc(uint32_t index) const
    {
        return index < attributes.size() ? &attributes[index] : nullptr;
    }

    FramebufferHandle Device::createFramebuffer(const FramebufferDesc& desc)
    {
        Framebuffer* framebuffer = new Framebuffer();
        framebuffer->desc = desc;
        framebuffer->framebufferInfo = FramebufferInfoEx(desc);
        return FramebufferHandle::Create(framebuffer);
    }

    static void applyColorBlend(MTLRenderPipelineColorAttachmentDescriptor* attachment, const BlendState::RenderTarget& blend)
    {
        attachment.blendingEnabled = blend.blendEnable;
        attachment.sourceRGBBlendFactor = convertBlendFactor(blend.srcBlend);
        attachment.destinationRGBBlendFactor = convertBlendFactor(blend.destBlend);
        attachment.rgbBlendOperation = convertBlendOp(blend.blendOp);
        attachment.sourceAlphaBlendFactor = convertBlendFactor(blend.srcBlendAlpha);
        attachment.destinationAlphaBlendFactor = convertBlendFactor(blend.destBlendAlpha);
        attachment.alphaBlendOperation = convertBlendOp(blend.blendOpAlpha);
        attachment.writeMask = MTLColorWriteMaskNone;
        if (!!(blend.colorWriteMask & ColorMask::Red)) attachment.writeMask |= MTLColorWriteMaskRed;
        if (!!(blend.colorWriteMask & ColorMask::Green)) attachment.writeMask |= MTLColorWriteMaskGreen;
        if (!!(blend.colorWriteMask & ColorMask::Blue)) attachment.writeMask |= MTLColorWriteMaskBlue;
        if (!!(blend.colorWriteMask & ColorMask::Alpha)) attachment.writeMask |= MTLColorWriteMaskAlpha;
    }

    static const char* shaderEntryName(const Shader* shader)
    {
        return shader && !shader->desc.entryName.empty() ? shader->desc.entryName.c_str() : "main";
    }

    static bool geometryInputMatchesPrimitive(const std::string& inputPrimitive, PrimitiveType primitiveType)
    {
        if (inputPrimitive == "Point")
            return primitiveType == PrimitiveType::PointList;
        if (inputPrimitive == "Line")
            return primitiveType == PrimitiveType::LineList || primitiveType == PrimitiveType::LineStrip;
        if (inputPrimitive == "Triangle")
            return primitiveType == PrimitiveType::TriangleList || primitiveType == PrimitiveType::TriangleStrip ||
                primitiveType == PrimitiveType::TriangleFan;
        if (inputPrimitive == "LineWithAdj")
            return false;
        if (inputPrimitive == "TriangleWithAdj")
            return primitiveType == PrimitiveType::TriangleListWithAdjacency ||
                primitiveType == PrimitiveType::TriangleStripWithAdjacency;

        return false;
    }

    static void applyFramebufferFormats(MTLRenderPipelineDescriptor* pd, FramebufferInfo const& fbinfo, const BlendState& blendState)
    {
        for (NSUInteger i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            pd.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);
            applyColorBlend(pd.colorAttachments[i], blendState.targets[i]);
        }
        if (fbinfo.depthFormat != Format::UNKNOWN)
            pd.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        if (getFormatInfo(fbinfo.depthFormat).hasStencil)
            pd.stencilAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        pd.rasterSampleCount = fbinfo.sampleCount;
    }

    static void applyFramebufferFormats(MTLMeshRenderPipelineDescriptor* pd, FramebufferInfo const& fbinfo, const BlendState& blendState)
    {
        for (NSUInteger i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            pd.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);
            applyColorBlend(pd.colorAttachments[i], blendState.targets[i]);
        }
        if (fbinfo.depthFormat != Format::UNKNOWN)
            pd.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        if (getFormatInfo(fbinfo.depthFormat).hasStencil)
            pd.stencilAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        pd.rasterSampleCount = fbinfo.sampleCount;
    }

    static bool validateGeometryEmulationShader(
        const MTL3Context& context,
        const GraphicsPipelineDesc& desc,
        const Shader* vs,
        const Shader* gs,
        const Shader* ps,
        IRRuntimeGeometryPipelineConfig& outConfig)
    {
        if (!vs || !gs || !ps)
        {
            context.error("[metal3] geometry emulation pipelines require vertex, geometry, and pixel shaders");
            return false;
        }
        if (!vs->library || !gs->library || !ps->library)
        {
            context.error("[metal3] geometry emulation pipeline has a shader without a Metal library");
            return false;
        }
        if (!vs->stageInLibrary)
        {
            context.error("[metal3] geometry emulation VS '" + vs->desc.debugName +
                "' has no MSC stage-in library");
            return false;
        }
        if (!vs->function || !gs->function || !ps->function)
        {
            context.error("[metal3] geometry emulation pipeline has a shader without a Metal entry function");
            return false;
        }
        if (vs->stageInLibrary.functionNames.count == 0)
        {
            context.error("[metal3] geometry emulation vertex shader has no stage-in function candidates");
            return false;
        }
        if (vs->mscReflection.vertexOutputSizeInBytes == 0)
        {
            context.error("[metal3] geometry emulation VS '" + vs->desc.debugName +
                "' has no reflected vertex_output_size_in_bytes");
            return false;
        }
        if (gs->mscReflection.maxInputPrimitivesPerMeshThreadgroup == 0)
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' has no reflected max_input_primitives_per_mesh_threadgroup");
            return false;
        }
        if (gs->mscReflection.inputPrimitive.empty())
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' has no reflected input_primitive");
            return false;
        }
        if (!geometryInputMatchesPrimitive(gs->mscReflection.inputPrimitive, desc.primType))
        {
            context.error("[metal3] geometry emulation GS '" + gs->desc.debugName +
                "' input primitive '" + gs->mscReflection.inputPrimitive +
                "' does not match the graphics pipeline primitive topology");
            return false;
        }

        outConfig.gsVertexSizeInBytes = vs->mscReflection.vertexOutputSizeInBytes;
        outConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
            gs->mscReflection.maxInputPrimitivesPerMeshThreadgroup;
        return true;
    }

    static id<MTLRenderPipelineState> createGeometryEmulationPipelineState(
        const MTL3Context& context,
        const GraphicsPipelineDesc& desc,
        FramebufferInfo const& fbinfo,
        Shader* vs,
        Shader* gs,
        Shader* ps,
        IRRuntimeGeometryPipelineConfig& outConfig)
    {
        if (!validateGeometryEmulationShader(context, desc, vs, gs, ps, outConfig))
            return nil;

        if (@available(macOS 14.0, *))
        {
            MTLMeshRenderPipelineDescriptor* pd = [[MTLMeshRenderPipelineDescriptor alloc] init];
            pd.label = @"nvrhi geometry emulation pipeline";
            pd.supportIndirectCommandBuffers = YES;
            applyFramebufferFormats(pd, fbinfo, desc.renderState.blendState);

            /* The MSC converter runtime creates the object/mesh functions from the
               converted VS/GS libraries, links the generated stage-in helper,
               and returns a regular MTLRenderPipelineState backed by a Metal
               mesh render pipeline. */
            const std::string vsEntry = shaderEntryName(vs);
            const std::string gsEntry = shaderEntryName(gs);
            const std::string psEntry = shaderEntryName(ps);

            IRGeometryEmulationPipelineDescriptor irDesc{};
            irDesc.stageInLibrary = vs->stageInLibrary;
            irDesc.vertexLibrary = vs->library;
            irDesc.vertexFunctionName = vsEntry.c_str();
            irDesc.geometryLibrary = gs->library;
            irDesc.geometryFunctionName = gsEntry.c_str();
            irDesc.fragmentLibrary = ps->library;
            irDesc.fragmentFunctionName = psEntry.c_str();
            irDesc.basePipelineDescriptor = pd;
            irDesc.pipelineConfig = outConfig;

            NSError* error = nil;
            id<MTLRenderPipelineState> nativePipeline =
                IRRuntimeNewGeometryEmulationPipeline(context.device, &irDesc, &error);
            if (!nativePipeline)
            {
                std::string message = "[nvrhi] Failed to create Metal geometry emulation pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                context.error(message);
                return nil;
            }

            return nativePipeline;
        }

        context.error("[metal3] geometry emulation pipelines require macOS 14.0 or newer");
        return nil;
    }

    static MTLStencilOperation convertStencilOp(StencilOp operation)
    {
        switch (operation)
        {
        case StencilOp::Keep: return MTLStencilOperationKeep;
        case StencilOp::Zero: return MTLStencilOperationZero;
        case StencilOp::Replace: return MTLStencilOperationReplace;
        case StencilOp::IncrementAndClamp: return MTLStencilOperationIncrementClamp;
        case StencilOp::DecrementAndClamp: return MTLStencilOperationDecrementClamp;
        case StencilOp::Invert: return MTLStencilOperationInvert;
        case StencilOp::IncrementAndWrap: return MTLStencilOperationIncrementWrap;
        case StencilOp::DecrementAndWrap: return MTLStencilOperationDecrementWrap;
        }
        return MTLStencilOperationKeep;
    }

    static MTLStencilDescriptor* createStencilDescriptor(const DepthStencilState::StencilOpDesc& face, const DepthStencilState& state)
    {
        MTLStencilDescriptor* descriptor = [[MTLStencilDescriptor alloc] init];
        descriptor.stencilCompareFunction = convertCompareFunction(face.stencilFunc);
        descriptor.stencilFailureOperation = convertStencilOp(face.failOp);
        descriptor.depthFailureOperation = convertStencilOp(face.depthFailOp);
        descriptor.depthStencilPassOperation = convertStencilOp(face.passOp);
        descriptor.readMask = state.stencilReadMask;
        descriptor.writeMask = state.stencilWriteMask;
        return descriptor;
    }

    static id<MTLDepthStencilState> createDepthStencilState(const MTL3Context& context, const DepthStencilState& state)
    {
        MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = state.depthTestEnable ? convertCompareFunction(state.depthFunc) : MTLCompareFunctionAlways;
        dd.depthWriteEnabled = state.depthWriteEnable;
        if (state.stencilEnable)
        {
            dd.frontFaceStencil = createStencilDescriptor(state.frontFaceStencil, state);
            dd.backFaceStencil = createStencilDescriptor(state.backFaceStencil, state);
        }
        return [context.device newDepthStencilStateWithDescriptor:dd];
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        auto* vs = static_cast<Shader*>(desc.VS.Get());
        auto* gs = static_cast<Shader*>(desc.GS.Get());
        auto* ps = static_cast<Shader*>(desc.PS.Get());
        if (!vs || !vs->function)
            return nullptr;

        const bool usesGeometryEmulation = gs != nullptr;
        IRRuntimeGeometryPipelineConfig geometryConfig{};
        NSError* error = nil;
        id<MTLRenderPipelineState> nativePipeline = nil;
        if (usesGeometryEmulation)
        {
            nativePipeline = createGeometryEmulationPipelineState(m_Context, desc, fbinfo, vs, gs, ps, geometryConfig);
            if (!nativePipeline)
                return nullptr;
        }
        else
        {
            MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
            pd.vertexFunction = vs->function;
            pd.fragmentFunction = ps ? ps->function : nil;  // can be nil for depth pass pipelines
            pd.supportIndirectCommandBuffers = YES;
            if (desc.inputLayout)
            {
                auto* inputLayout = static_cast<InputLayout*>(desc.inputLayout.Get());
                pd.vertexDescriptor = inputLayout ? inputLayout->vertexDescriptor : nil;
            }

            applyFramebufferFormats(pd, fbinfo, desc.renderState.blendState);
            nativePipeline = createRenderPipelineState(pd, &error);
            if (!nativePipeline)
            {
                std::string message = "[nvrhi] Failed to create Metal render pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                m_Context.error(message);
                return nullptr;
            }
        }

        GraphicsPipeline* pipeline = new GraphicsPipeline();
        pipeline->desc = desc;
        pipeline->framebufferInfo = fbinfo;
        pipeline->pipeline = nativePipeline;
        pipeline->depthStencilState = createDepthStencilState(m_Context, desc.renderState.depthStencilState);
        pipeline->primitiveType = convertPrimitiveType(desc.primType);
        pipeline->cullMode = convertCullMode(desc.renderState.rasterState.cullMode);
        pipeline->frontWinding = convertWinding(desc.renderState.rasterState.frontCounterClockwise);
        pipeline->usesGeometryEmulation = usesGeometryEmulation;
        if (pipeline->usesGeometryEmulation)
        {
            pipeline->geometryVertexSizeInBytes = geometryConfig.gsVertexSizeInBytes;
            pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup =
                geometryConfig.gsMaxInputPrimitivesPerMeshThreadgroup;
            pipeline->geometryInstanceCount = gs->mscReflection.geometryInstanceCount;
            pipeline->objectBindingPlan = resolveMetalStageBindingPlan(vs->reflectedBindingPlan, desc.bindingLayouts);
            pipeline->meshBindingPlan = resolveMetalStageBindingPlan(gs->reflectedBindingPlan, desc.bindingLayouts);
        }

        // create vertex and fragment plan
        pipeline->vertexBindingPlan = resolveMetalStageBindingPlan(vs->reflectedBindingPlan, desc.bindingLayouts);
        if (ps)
            pipeline->fragmentBindingPlan = resolveMetalStageBindingPlan(ps->reflectedBindingPlan, desc.bindingLayouts);
        return GraphicsPipelineHandle::Create(pipeline);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb)
    {
        return createGraphicsPipeline(desc, fb ? FramebufferInfo(fb->getDesc()) : FramebufferInfo());
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        auto* as = static_cast<Shader*>(desc.AS.Get());
        auto* ms = static_cast<Shader*>(desc.MS.Get());
        auto* ps = static_cast<Shader*>(desc.PS.Get());
        if (!ms || !ms->function || (as && !as->function))
        {
            m_Context.error("[nvrhi] Metal meshlet pipelines require a mesh shader function.");
            return nullptr;
        }
        if (!ms->computeThreadsPerGroupValid || (as && !as->computeThreadsPerGroupValid))
        {
            m_Context.error("[nvrhi] Metal meshlet pipelines require reflected thread counts for the mesh and amplification shaders.");
            return nullptr;
        }

        id<MTLRenderPipelineState> nativePipeline = nil;
        if (@available(macOS 14.0, *))
        {
            MTLMeshRenderPipelineDescriptor* pd = [[MTLMeshRenderPipelineDescriptor alloc] init];
            pd.objectFunction = as ? as->function : nil;
            pd.meshFunction = ms->function;
            pd.fragmentFunction = ps ? ps->function : nil;
            pd.supportIndirectCommandBuffers = YES;
            pd.maxTotalThreadsPerMeshThreadgroup = ms->computeThreadsPerGroup.width *
                ms->computeThreadsPerGroup.height * ms->computeThreadsPerGroup.depth;
            if (as)
            {
                pd.maxTotalThreadsPerObjectThreadgroup = as->computeThreadsPerGroup.width *
                    as->computeThreadsPerGroup.height * as->computeThreadsPerGroup.depth;
                uint32_t payload = std::max(as->mscReflection.maxPayloadSizeInBytes, ms->mscReflection.maxPayloadSizeInBytes);
                pd.payloadMemoryLength = payload ? payload : 16384;
            }
            applyFramebufferFormats(pd, fbinfo, desc.renderState.blendState);
            NSError* error = nil;
            nativePipeline = createMeshRenderPipelineState(pd, &error);
            if (!nativePipeline)
            {
                std::string message = "[nvrhi] Failed to create Metal mesh render pipeline";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                m_Context.error(message);
                return nullptr;
            }
        }
        else
        {
            m_Context.error("[nvrhi] Metal meshlet pipelines require macOS 14.0 or newer.");
            return nullptr;
        }

        MeshletPipeline* pipeline = new MeshletPipeline();
        pipeline->desc = desc;
        pipeline->framebufferInfo = fbinfo;
        pipeline->pipeline = nativePipeline;
        pipeline->depthStencilState = createDepthStencilState(m_Context, desc.renderState.depthStencilState);
        pipeline->cullMode = convertCullMode(desc.renderState.rasterState.cullMode);
        pipeline->frontWinding = convertWinding(desc.renderState.rasterState.frontCounterClockwise);
        pipeline->hasObjectStage = as != nullptr;
        pipeline->threadsPerMeshThreadgroup = ms->computeThreadsPerGroup;
        if (as)
        {
            pipeline->threadsPerObjectThreadgroup = as->computeThreadsPerGroup;
            pipeline->objectBindingPlan = resolveMetalStageBindingPlan(as->reflectedBindingPlan, desc.bindingLayouts);
        }
        pipeline->meshBindingPlan = resolveMetalStageBindingPlan(ms->reflectedBindingPlan, desc.bindingLayouts);
        if (ps)
            pipeline->fragmentBindingPlan = resolveMetalStageBindingPlan(ps->reflectedBindingPlan, desc.bindingLayouts);
        return MeshletPipelineHandle::Create(pipeline);
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb)
    {
        return createMeshletPipeline(desc, fb ? FramebufferInfo(fb->getDesc()) : FramebufferInfo());
    }

    Object GraphicsPipeline::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_RenderPipeline)
            return Object((__bridge void*)pipeline);
        return nullptr;
    }
}
