/*
 * Copyright 2018-2021 Arm Limited
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

#include "spirv_parser.hpp"
#include "spirv_common.hpp"
#include "NonSemanticShaderDebugInfo100.h"
#include <assert.h>

using namespace std;
using namespace SPIRV_CROSS_SPV_HEADER_NAMESPACE;

namespace SPIRV_CROSS_NAMESPACE
{
Parser::Parser(vector<uint32_t> spirv)
{
	ir.spirv = std::move(spirv);
}

Parser::Parser(const uint32_t *spirv_data, size_t word_count)
{
	ir.spirv = vector<uint32_t>(spirv_data, spirv_data + word_count);
}

static bool decoration_is_string(Decoration decoration)
{
	switch (decoration)
	{
	case DecorationUserSemantic:
		return true;

	default:
		return false;
	}
}

static inline uint32_t swap_endian(uint32_t v)
{
	return ((v >> 24) & 0x000000ffu) | ((v >> 8) & 0x0000ff00u) | ((v << 8) & 0x00ff0000u) | ((v << 24) & 0xff000000u);
}

static bool is_valid_spirv_version(uint32_t version)
{
	switch (version)
	{
	// Allow v99 since it tends to just work.
	case 99:
	case 0x10000: // SPIR-V 1.0
	case 0x10100: // SPIR-V 1.1
	case 0x10200: // SPIR-V 1.2
	case 0x10300: // SPIR-V 1.3
	case 0x10400: // SPIR-V 1.4
	case 0x10500: // SPIR-V 1.5
	case 0x10600: // SPIR-V 1.6
		return true;

	default:
		return false;
	}
}

// Enumeration validators. Malformed modules can place arbitrary bit patterns in
// enum-typed operands; storing those into enum-typed fields is undefined behavior
// and later loads crash under UBSan or enum-hardened runtimes. Validate every
// enum operand against the enumerators known to this header before casting.

// Returns true if the literal is a valid Op enumerator known to this header.
static bool is_known_opcode(uint32_t v)
{
	switch (v)
	{
	case OpNop:
	case OpUndef:
	case OpSourceContinued:
	case OpSource:
	case OpSourceExtension:
	case OpName:
	case OpMemberName:
	case OpString:
	case OpLine:
	case OpExtension:
	case OpExtInstImport:
	case OpExtInst:
	case OpMemoryModel:
	case OpEntryPoint:
	case OpExecutionMode:
	case OpCapability:
	case OpTypeVoid:
	case OpTypeBool:
	case OpTypeInt:
	case OpTypeFloat:
	case OpTypeVector:
	case OpTypeMatrix:
	case OpTypeImage:
	case OpTypeSampler:
	case OpTypeSampledImage:
	case OpTypeArray:
	case OpTypeRuntimeArray:
	case OpTypeStruct:
	case OpTypeOpaque:
	case OpTypePointer:
	case OpTypeFunction:
	case OpTypeEvent:
	case OpTypeDeviceEvent:
	case OpTypeReserveId:
	case OpTypeQueue:
	case OpTypePipe:
	case OpTypeForwardPointer:
	case OpConstantTrue:
	case OpConstantFalse:
	case OpConstant:
	case OpConstantComposite:
	case OpConstantSampler:
	case OpConstantNull:
	case OpSpecConstantTrue:
	case OpSpecConstantFalse:
	case OpSpecConstant:
	case OpSpecConstantComposite:
	case OpSpecConstantOp:
	case OpFunction:
	case OpFunctionParameter:
	case OpFunctionEnd:
	case OpFunctionCall:
	case OpVariable:
	case OpImageTexelPointer:
	case OpLoad:
	case OpStore:
	case OpCopyMemory:
	case OpCopyMemorySized:
	case OpAccessChain:
	case OpInBoundsAccessChain:
	case OpPtrAccessChain:
	case OpArrayLength:
	case OpGenericPtrMemSemantics:
	case OpInBoundsPtrAccessChain:
	case OpDecorate:
	case OpMemberDecorate:
	case OpDecorationGroup:
	case OpGroupDecorate:
	case OpGroupMemberDecorate:
	case OpVectorExtractDynamic:
	case OpVectorInsertDynamic:
	case OpVectorShuffle:
	case OpCompositeConstruct:
	case OpCompositeExtract:
	case OpCompositeInsert:
	case OpCopyObject:
	case OpTranspose:
	case OpSampledImage:
	case OpImageSampleImplicitLod:
	case OpImageSampleExplicitLod:
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleDrefExplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageSampleProjExplicitLod:
	case OpImageSampleProjDrefImplicitLod:
	case OpImageSampleProjDrefExplicitLod:
	case OpImageFetch:
	case OpImageGather:
	case OpImageDrefGather:
	case OpImageRead:
	case OpImageWrite:
	case OpImage:
	case OpImageQueryFormat:
	case OpImageQueryOrder:
	case OpImageQuerySizeLod:
	case OpImageQuerySize:
	case OpImageQueryLod:
	case OpImageQueryLevels:
	case OpImageQuerySamples:
	case OpConvertFToU:
	case OpConvertFToS:
	case OpConvertSToF:
	case OpConvertUToF:
	case OpUConvert:
	case OpSConvert:
	case OpFConvert:
	case OpQuantizeToF16:
	case OpConvertPtrToU:
	case OpSatConvertSToU:
	case OpSatConvertUToS:
	case OpConvertUToPtr:
	case OpPtrCastToGeneric:
	case OpGenericCastToPtr:
	case OpGenericCastToPtrExplicit:
	case OpBitcast:
	case OpSNegate:
	case OpFNegate:
	case OpIAdd:
	case OpFAdd:
	case OpISub:
	case OpFSub:
	case OpIMul:
	case OpFMul:
	case OpUDiv:
	case OpSDiv:
	case OpFDiv:
	case OpUMod:
	case OpSRem:
	case OpSMod:
	case OpFRem:
	case OpFMod:
	case OpVectorTimesScalar:
	case OpMatrixTimesScalar:
	case OpVectorTimesMatrix:
	case OpMatrixTimesVector:
	case OpMatrixTimesMatrix:
	case OpOuterProduct:
	case OpDot:
	case OpIAddCarry:
	case OpISubBorrow:
	case OpUMulExtended:
	case OpSMulExtended:
	case OpAny:
	case OpAll:
	case OpIsNan:
	case OpIsInf:
	case OpIsFinite:
	case OpIsNormal:
	case OpSignBitSet:
	case OpLessOrGreater:
	case OpOrdered:
	case OpUnordered:
	case OpLogicalEqual:
	case OpLogicalNotEqual:
	case OpLogicalOr:
	case OpLogicalAnd:
	case OpLogicalNot:
	case OpSelect:
	case OpIEqual:
	case OpINotEqual:
	case OpUGreaterThan:
	case OpSGreaterThan:
	case OpUGreaterThanEqual:
	case OpSGreaterThanEqual:
	case OpULessThan:
	case OpSLessThan:
	case OpULessThanEqual:
	case OpSLessThanEqual:
	case OpFOrdEqual:
	case OpFUnordEqual:
	case OpFOrdNotEqual:
	case OpFUnordNotEqual:
	case OpFOrdLessThan:
	case OpFUnordLessThan:
	case OpFOrdGreaterThan:
	case OpFUnordGreaterThan:
	case OpFOrdLessThanEqual:
	case OpFUnordLessThanEqual:
	case OpFOrdGreaterThanEqual:
	case OpFUnordGreaterThanEqual:
	case OpShiftRightLogical:
	case OpShiftRightArithmetic:
	case OpShiftLeftLogical:
	case OpBitwiseOr:
	case OpBitwiseXor:
	case OpBitwiseAnd:
	case OpNot:
	case OpBitFieldInsert:
	case OpBitFieldSExtract:
	case OpBitFieldUExtract:
	case OpBitReverse:
	case OpBitCount:
	case OpDPdx:
	case OpDPdy:
	case OpFwidth:
	case OpDPdxFine:
	case OpDPdyFine:
	case OpFwidthFine:
	case OpDPdxCoarse:
	case OpDPdyCoarse:
	case OpFwidthCoarse:
	case OpEmitVertex:
	case OpEndPrimitive:
	case OpEmitStreamVertex:
	case OpEndStreamPrimitive:
	case OpControlBarrier:
	case OpMemoryBarrier:
	case OpAtomicLoad:
	case OpAtomicStore:
	case OpAtomicExchange:
	case OpAtomicCompareExchange:
	case OpAtomicCompareExchangeWeak:
	case OpAtomicIIncrement:
	case OpAtomicIDecrement:
	case OpAtomicIAdd:
	case OpAtomicISub:
	case OpAtomicSMin:
	case OpAtomicUMin:
	case OpAtomicSMax:
	case OpAtomicUMax:
	case OpAtomicAnd:
	case OpAtomicOr:
	case OpAtomicXor:
	case OpPhi:
	case OpLoopMerge:
	case OpSelectionMerge:
	case OpLabel:
	case OpBranch:
	case OpBranchConditional:
	case OpSwitch:
	case OpKill:
	case OpReturn:
	case OpReturnValue:
	case OpUnreachable:
	case OpLifetimeStart:
	case OpLifetimeStop:
	case OpGroupAsyncCopy:
	case OpGroupWaitEvents:
	case OpGroupAll:
	case OpGroupAny:
	case OpGroupBroadcast:
	case OpGroupIAdd:
	case OpGroupFAdd:
	case OpGroupFMin:
	case OpGroupUMin:
	case OpGroupSMin:
	case OpGroupFMax:
	case OpGroupUMax:
	case OpGroupSMax:
	case OpReadPipe:
	case OpWritePipe:
	case OpReservedReadPipe:
	case OpReservedWritePipe:
	case OpReserveReadPipePackets:
	case OpReserveWritePipePackets:
	case OpCommitReadPipe:
	case OpCommitWritePipe:
	case OpIsValidReserveId:
	case OpGetNumPipePackets:
	case OpGetMaxPipePackets:
	case OpGroupReserveReadPipePackets:
	case OpGroupReserveWritePipePackets:
	case OpGroupCommitReadPipe:
	case OpGroupCommitWritePipe:
	case OpEnqueueMarker:
	case OpEnqueueKernel:
	case OpGetKernelNDrangeSubGroupCount:
	case OpGetKernelNDrangeMaxSubGroupSize:
	case OpGetKernelWorkGroupSize:
	case OpGetKernelPreferredWorkGroupSizeMultiple:
	case OpRetainEvent:
	case OpReleaseEvent:
	case OpCreateUserEvent:
	case OpIsValidEvent:
	case OpSetUserEventStatus:
	case OpCaptureEventProfilingInfo:
	case OpGetDefaultQueue:
	case OpBuildNDRange:
	case OpImageSparseSampleImplicitLod:
	case OpImageSparseSampleExplicitLod:
	case OpImageSparseSampleDrefImplicitLod:
	case OpImageSparseSampleDrefExplicitLod:
	case OpImageSparseSampleProjImplicitLod:
	case OpImageSparseSampleProjExplicitLod:
	case OpImageSparseSampleProjDrefImplicitLod:
	case OpImageSparseSampleProjDrefExplicitLod:
	case OpImageSparseFetch:
	case OpImageSparseGather:
	case OpImageSparseDrefGather:
	case OpImageSparseTexelsResident:
	case OpNoLine:
	case OpAtomicFlagTestAndSet:
	case OpAtomicFlagClear:
	case OpImageSparseRead:
	case OpSizeOf:
	case OpTypePipeStorage:
	case OpConstantPipeStorage:
	case OpCreatePipeFromPipeStorage:
	case OpGetKernelLocalSizeForSubgroupCount:
	case OpGetKernelMaxNumSubgroups:
	case OpTypeNamedBarrier:
	case OpNamedBarrierInitialize:
	case OpMemoryNamedBarrier:
	case OpModuleProcessed:
	case OpExecutionModeId:
	case OpDecorateId:
	case OpGroupNonUniformElect:
	case OpGroupNonUniformAll:
	case OpGroupNonUniformAny:
	case OpGroupNonUniformAllEqual:
	case OpGroupNonUniformBroadcast:
	case OpGroupNonUniformBroadcastFirst:
	case OpGroupNonUniformBallot:
	case OpGroupNonUniformInverseBallot:
	case OpGroupNonUniformBallotBitExtract:
	case OpGroupNonUniformBallotBitCount:
	case OpGroupNonUniformBallotFindLSB:
	case OpGroupNonUniformBallotFindMSB:
	case OpGroupNonUniformShuffle:
	case OpGroupNonUniformShuffleXor:
	case OpGroupNonUniformShuffleUp:
	case OpGroupNonUniformShuffleDown:
	case OpGroupNonUniformIAdd:
	case OpGroupNonUniformFAdd:
	case OpGroupNonUniformIMul:
	case OpGroupNonUniformFMul:
	case OpGroupNonUniformSMin:
	case OpGroupNonUniformUMin:
	case OpGroupNonUniformFMin:
	case OpGroupNonUniformSMax:
	case OpGroupNonUniformUMax:
	case OpGroupNonUniformFMax:
	case OpGroupNonUniformBitwiseAnd:
	case OpGroupNonUniformBitwiseOr:
	case OpGroupNonUniformBitwiseXor:
	case OpGroupNonUniformLogicalAnd:
	case OpGroupNonUniformLogicalOr:
	case OpGroupNonUniformLogicalXor:
	case OpGroupNonUniformQuadBroadcast:
	case OpGroupNonUniformQuadSwap:
	case OpCopyLogical:
	case OpPtrEqual:
	case OpPtrNotEqual:
	case OpPtrDiff:
	case OpColorAttachmentReadEXT:
	case OpDepthAttachmentReadEXT:
	case OpStencilAttachmentReadEXT:
	case OpTypeTensorARM:
	case OpTensorReadARM:
	case OpTensorWriteARM:
	case OpTensorQuerySizeARM:
	case OpGraphConstantARM:
	case OpGraphEntryPointARM:
	case OpGraphARM:
	case OpGraphInputARM:
	case OpGraphSetOutputARM:
	case OpGraphEndARM:
	case OpTypeGraphARM:
	case OpTerminateInvocation:
	case OpTypeUntypedPointerKHR:
	case OpUntypedVariableKHR:
	case OpUntypedAccessChainKHR:
	case OpUntypedInBoundsAccessChainKHR:
	case OpSubgroupBallotKHR:
	case OpSubgroupFirstInvocationKHR:
	case OpUntypedPtrAccessChainKHR:
	case OpUntypedInBoundsPtrAccessChainKHR:
	case OpUntypedArrayLengthKHR:
	case OpUntypedPrefetchKHR:
	case OpFmaKHR:
	case OpSubgroupAllKHR:
	case OpSubgroupAnyKHR:
	case OpSubgroupAllEqualKHR:
	case OpGroupNonUniformRotateKHR:
	case OpSubgroupReadInvocationKHR:
	case OpExtInstWithForwardRefsKHR:
	case OpUntypedGroupAsyncCopyKHR:
	case OpTraceRayKHR:
	case OpExecuteCallableKHR:
	case OpConvertUToAccelerationStructureKHR:
	case OpIgnoreIntersectionKHR:
	case OpTerminateRayKHR:
	case OpSDot:
	case OpUDot:
	case OpSUDot:
	case OpSDotAccSat:
	case OpUDotAccSat:
	case OpSUDotAccSat:
	case OpTypeCooperativeMatrixKHR:
	case OpCooperativeMatrixLoadKHR:
	case OpCooperativeMatrixStoreKHR:
	case OpCooperativeMatrixMulAddKHR:
	case OpCooperativeMatrixLengthKHR:
	case OpConstantCompositeReplicateEXT:
	case OpSpecConstantCompositeReplicateEXT:
	case OpCompositeConstructReplicateEXT:
	case OpTypeRayQueryKHR:
	case OpRayQueryInitializeKHR:
	case OpRayQueryTerminateKHR:
	case OpRayQueryGenerateIntersectionKHR:
	case OpRayQueryConfirmIntersectionKHR:
	case OpRayQueryProceedKHR:
	case OpRayQueryGetIntersectionTypeKHR:
	case OpImageSampleWeightedQCOM:
	case OpImageBoxFilterQCOM:
	case OpImageBlockMatchSSDQCOM:
	case OpImageBlockMatchSADQCOM:
	case OpBitCastArrayQCOM:
	case OpImageBlockMatchWindowSSDQCOM:
	case OpImageBlockMatchWindowSADQCOM:
	case OpImageBlockMatchGatherSSDQCOM:
	case OpImageBlockMatchGatherSADQCOM:
	case OpCompositeConstructCoopMatQCOM:
	case OpCompositeExtractCoopMatQCOM:
	case OpExtractSubArrayQCOM:
	case OpGroupIAddNonUniformAMD:
	case OpGroupFAddNonUniformAMD:
	case OpGroupFMinNonUniformAMD:
	case OpGroupUMinNonUniformAMD:
	case OpGroupSMinNonUniformAMD:
	case OpGroupFMaxNonUniformAMD:
	case OpGroupUMaxNonUniformAMD:
	case OpGroupSMaxNonUniformAMD:
	case OpFragmentMaskFetchAMD:
	case OpFragmentFetchAMD:
	case OpReadClockKHR:
	case OpAllocateNodePayloadsAMDX:
	case OpEnqueueNodePayloadsAMDX:
	case OpTypeNodePayloadArrayAMDX:
	case OpFinishWritingNodePayloadAMDX:
	case OpNodePayloadArrayLengthAMDX:
	case OpIsNodePayloadValidAMDX:
	case OpConstantStringAMDX:
	case OpSpecConstantStringAMDX:
	case OpGroupNonUniformQuadAllKHR:
	case OpGroupNonUniformQuadAnyKHR:
	case OpTypeBufferEXT:
	case OpBufferPointerEXT:
	case OpAbortKHR:
	case OpUntypedImageTexelPointerEXT:
	case OpMemberDecorateIdEXT:
	case OpConstantSizeOfEXT:
	case OpConstantDataKHR:
	case OpSpecConstantDataKHR:
	case OpPoisonKHR:
	case OpFreezeKHR:
	case OpHitObjectRecordHitMotionNV:
	case OpHitObjectRecordHitWithIndexMotionNV:
	case OpHitObjectRecordMissMotionNV:
	case OpHitObjectGetWorldToObjectNV:
	case OpHitObjectGetObjectToWorldNV:
	case OpHitObjectGetObjectRayDirectionNV:
	case OpHitObjectGetObjectRayOriginNV:
	case OpHitObjectTraceRayMotionNV:
	case OpHitObjectGetShaderRecordBufferHandleNV:
	case OpHitObjectGetShaderBindingTableRecordIndexNV:
	case OpHitObjectRecordEmptyNV:
	case OpHitObjectTraceRayNV:
	case OpHitObjectRecordHitNV:
	case OpHitObjectRecordHitWithIndexNV:
	case OpHitObjectRecordMissNV:
	case OpHitObjectExecuteShaderNV:
	case OpHitObjectGetCurrentTimeNV:
	case OpHitObjectGetAttributesNV:
	case OpHitObjectGetHitKindNV:
	case OpHitObjectGetPrimitiveIndexNV:
	case OpHitObjectGetGeometryIndexNV:
	case OpHitObjectGetInstanceIdNV:
	case OpHitObjectGetInstanceCustomIndexNV:
	case OpHitObjectGetWorldRayDirectionNV:
	case OpHitObjectGetWorldRayOriginNV:
	case OpHitObjectGetRayTMaxNV:
	case OpHitObjectGetRayTMinNV:
	case OpHitObjectIsEmptyNV:
	case OpHitObjectIsHitNV:
	case OpHitObjectIsMissNV:
	case OpReorderThreadWithHitObjectNV:
	case OpReorderThreadWithHintNV:
	case OpTypeHitObjectNV:
	case OpImageSampleFootprintNV:
	case OpTypeCooperativeVectorNV:
	case OpCooperativeVectorMatrixMulNV:
	case OpCooperativeVectorOuterProductAccumulateNV:
	case OpCooperativeVectorReduceSumAccumulateNV:
	case OpCooperativeVectorMatrixMulAddNV:
	case OpCooperativeMatrixConvertNV:
	case OpEmitMeshTasksEXT:
	case OpSetMeshOutputsEXT:
	case OpGroupNonUniformPartitionEXT:
	case OpWritePackedPrimitiveIndices4x8NV:
	case OpFetchMicroTriangleVertexPositionNV:
	case OpFetchMicroTriangleVertexBarycentricNV:
	case OpCooperativeVectorLoadNV:
	case OpCooperativeVectorStoreNV:
	case OpHitObjectRecordFromQueryEXT:
	case OpHitObjectRecordMissEXT:
	case OpHitObjectRecordMissMotionEXT:
	case OpHitObjectGetIntersectionTriangleVertexPositionsEXT:
	case OpHitObjectGetRayFlagsEXT:
	case OpHitObjectSetShaderBindingTableRecordIndexEXT:
	case OpHitObjectReorderExecuteShaderEXT:
	case OpHitObjectTraceReorderExecuteEXT:
	case OpHitObjectTraceMotionReorderExecuteEXT:
	case OpTypeHitObjectEXT:
	case OpReorderThreadWithHintEXT:
	case OpReorderThreadWithHitObjectEXT:
	case OpHitObjectTraceRayEXT:
	case OpHitObjectTraceRayMotionEXT:
	case OpHitObjectRecordEmptyEXT:
	case OpHitObjectExecuteShaderEXT:
	case OpHitObjectGetCurrentTimeEXT:
	case OpHitObjectGetAttributesEXT:
	case OpHitObjectGetHitKindEXT:
	case OpHitObjectGetPrimitiveIndexEXT:
	case OpHitObjectGetGeometryIndexEXT:
	case OpHitObjectGetInstanceIdEXT:
	case OpHitObjectGetInstanceCustomIndexEXT:
	case OpHitObjectGetObjectRayOriginEXT:
	case OpHitObjectGetObjectRayDirectionEXT:
	case OpHitObjectGetWorldRayDirectionEXT:
	case OpHitObjectGetWorldRayOriginEXT:
	case OpHitObjectGetObjectToWorldEXT:
	case OpHitObjectGetWorldToObjectEXT:
	case OpHitObjectGetRayTMaxEXT:
	case OpReportIntersectionKHR:
	case OpIgnoreIntersectionNV:
	case OpTerminateRayNV:
	case OpTraceNV:
	case OpTraceMotionNV:
	case OpTraceRayMotionNV:
	case OpRayQueryGetIntersectionTriangleVertexPositionsKHR:
	case OpTypeAccelerationStructureKHR:
	case OpExecuteCallableNV:
	case OpRayQueryGetClusterIdNV:
	case OpHitObjectGetClusterIdNV:
	case OpHitObjectGetRayTMinEXT:
	case OpHitObjectGetShaderBindingTableRecordIndexEXT:
	case OpHitObjectGetShaderRecordBufferHandleEXT:
	case OpHitObjectIsEmptyEXT:
	case OpHitObjectIsHitEXT:
	case OpHitObjectIsMissEXT:
	case OpTypeCooperativeMatrixNV:
	case OpCooperativeMatrixLoadNV:
	case OpCooperativeMatrixStoreNV:
	case OpCooperativeMatrixMulAddNV:
	case OpCooperativeMatrixLengthNV:
	case OpBeginInvocationInterlockEXT:
	case OpEndInvocationInterlockEXT:
	case OpCooperativeMatrixReduceNV:
	case OpCooperativeMatrixLoadTensorNV:
	case OpCooperativeMatrixStoreTensorNV:
	case OpCooperativeMatrixPerElementOpNV:
	case OpTypeTensorLayoutNV:
	case OpTypeTensorViewNV:
	case OpCreateTensorLayoutNV:
	case OpTensorLayoutSetDimensionNV:
	case OpTensorLayoutSetStrideNV:
	case OpTensorLayoutSliceNV:
	case OpTensorLayoutSetClampValueNV:
	case OpCreateTensorViewNV:
	case OpTensorViewSetDimensionNV:
	case OpTensorViewSetStrideNV:
	case OpDemoteToHelperInvocation:
	case OpIsHelperInvocationEXT:
	case OpTensorViewSetClipNV:
	case OpTensorLayoutSetBlockSizeNV:
	case OpCooperativeMatrixTransposeNV:
	case OpConvertUToImageNV:
	case OpConvertUToSamplerNV:
	case OpConvertImageToUNV:
	case OpConvertSamplerToUNV:
	case OpConvertUToSampledImageNV:
	case OpConvertSampledImageToUNV:
	case OpSamplerImageAddressingModeNV:
	case OpRawAccessChainNV:
	case OpRayQueryGetIntersectionSpherePositionNV:
	case OpRayQueryGetIntersectionSphereRadiusNV:
	case OpRayQueryGetIntersectionLSSPositionsNV:
	case OpRayQueryGetIntersectionLSSRadiiNV:
	case OpRayQueryGetIntersectionLSSHitValueNV:
	case OpHitObjectGetSpherePositionNV:
	case OpHitObjectGetSphereRadiusNV:
	case OpHitObjectGetLSSPositionsNV:
	case OpHitObjectGetLSSRadiiNV:
	case OpHitObjectIsSphereHitNV:
	case OpHitObjectIsLSSHitNV:
	case OpRayQueryIsSphereHitNV:
	case OpRayQueryIsLSSHitNV:
	case OpSubgroupShuffleINTEL:
	case OpSubgroupShuffleDownINTEL:
	case OpSubgroupShuffleUpINTEL:
	case OpSubgroupShuffleXorINTEL:
	case OpSubgroupBlockReadINTEL:
	case OpSubgroupBlockWriteINTEL:
	case OpSubgroupImageBlockReadINTEL:
	case OpSubgroupImageBlockWriteINTEL:
	case OpSubgroupImageMediaBlockReadINTEL:
	case OpSubgroupImageMediaBlockWriteINTEL:
	case OpUCountLeadingZerosINTEL:
	case OpUCountTrailingZerosINTEL:
	case OpAbsISubINTEL:
	case OpAbsUSubINTEL:
	case OpIAddSatINTEL:
	case OpUAddSatINTEL:
	case OpIAverageINTEL:
	case OpUAverageINTEL:
	case OpIAverageRoundedINTEL:
	case OpUAverageRoundedINTEL:
	case OpISubSatINTEL:
	case OpUSubSatINTEL:
	case OpIMul32x16INTEL:
	case OpUMul32x16INTEL:
	case OpConstantFunctionPointerINTEL:
	case OpFunctionPointerCallINTEL:
	case OpAsmTargetINTEL:
	case OpAsmINTEL:
	case OpAsmCallINTEL:
	case OpAtomicFMinEXT:
	case OpAtomicFMaxEXT:
	case OpAssumeTrueKHR:
	case OpExpectKHR:
	case OpDecorateString:
	case OpMemberDecorateString:
	case OpVmeImageINTEL:
	case OpTypeVmeImageINTEL:
	case OpTypeAvcImePayloadINTEL:
	case OpTypeAvcRefPayloadINTEL:
	case OpTypeAvcSicPayloadINTEL:
	case OpTypeAvcMcePayloadINTEL:
	case OpTypeAvcMceResultINTEL:
	case OpTypeAvcImeResultINTEL:
	case OpTypeAvcImeResultSingleReferenceStreamoutINTEL:
	case OpTypeAvcImeResultDualReferenceStreamoutINTEL:
	case OpTypeAvcImeSingleReferenceStreaminINTEL:
	case OpTypeAvcImeDualReferenceStreaminINTEL:
	case OpTypeAvcRefResultINTEL:
	case OpTypeAvcSicResultINTEL:
	case OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL:
	case OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL:
	case OpSubgroupAvcMceSetInterShapePenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL:
	case OpSubgroupAvcMceSetInterDirectionPenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL:
	case OpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL:
	case OpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL:
	case OpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL:
	case OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL:
	case OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL:
	case OpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL:
	case OpSubgroupAvcMceSetAcOnlyHaarINTEL:
	case OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL:
	case OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL:
	case OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL:
	case OpSubgroupAvcMceConvertToImePayloadINTEL:
	case OpSubgroupAvcMceConvertToImeResultINTEL:
	case OpSubgroupAvcMceConvertToRefPayloadINTEL:
	case OpSubgroupAvcMceConvertToRefResultINTEL:
	case OpSubgroupAvcMceConvertToSicPayloadINTEL:
	case OpSubgroupAvcMceConvertToSicResultINTEL:
	case OpSubgroupAvcMceGetMotionVectorsINTEL:
	case OpSubgroupAvcMceGetInterDistortionsINTEL:
	case OpSubgroupAvcMceGetBestInterDistortionsINTEL:
	case OpSubgroupAvcMceGetInterMajorShapeINTEL:
	case OpSubgroupAvcMceGetInterMinorShapeINTEL:
	case OpSubgroupAvcMceGetInterDirectionsINTEL:
	case OpSubgroupAvcMceGetInterMotionVectorCountINTEL:
	case OpSubgroupAvcMceGetInterReferenceIdsINTEL:
	case OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL:
	case OpSubgroupAvcImeInitializeINTEL:
	case OpSubgroupAvcImeSetSingleReferenceINTEL:
	case OpSubgroupAvcImeSetDualReferenceINTEL:
	case OpSubgroupAvcImeRefWindowSizeINTEL:
	case OpSubgroupAvcImeAdjustRefOffsetINTEL:
	case OpSubgroupAvcImeConvertToMcePayloadINTEL:
	case OpSubgroupAvcImeSetMaxMotionVectorCountINTEL:
	case OpSubgroupAvcImeSetUnidirectionalMixDisableINTEL:
	case OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL:
	case OpSubgroupAvcImeSetWeightedSadINTEL:
	case OpSubgroupAvcImeEvaluateWithSingleReferenceINTEL:
	case OpSubgroupAvcImeEvaluateWithDualReferenceINTEL:
	case OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL:
	case OpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL:
	case OpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL:
	case OpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL:
	case OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL:
	case OpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL:
	case OpSubgroupAvcImeConvertToMceResultINTEL:
	case OpSubgroupAvcImeGetSingleReferenceStreaminINTEL:
	case OpSubgroupAvcImeGetDualReferenceStreaminINTEL:
	case OpSubgroupAvcImeStripSingleReferenceStreamoutINTEL:
	case OpSubgroupAvcImeStripDualReferenceStreamoutINTEL:
	case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL:
	case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL:
	case OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL:
	case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL:
	case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL:
	case OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL:
	case OpSubgroupAvcImeGetBorderReachedINTEL:
	case OpSubgroupAvcImeGetTruncatedSearchIndicationINTEL:
	case OpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL:
	case OpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL:
	case OpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL:
	case OpSubgroupAvcFmeInitializeINTEL:
	case OpSubgroupAvcBmeInitializeINTEL:
	case OpSubgroupAvcRefConvertToMcePayloadINTEL:
	case OpSubgroupAvcRefSetBidirectionalMixDisableINTEL:
	case OpSubgroupAvcRefSetBilinearFilterEnableINTEL:
	case OpSubgroupAvcRefEvaluateWithSingleReferenceINTEL:
	case OpSubgroupAvcRefEvaluateWithDualReferenceINTEL:
	case OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL:
	case OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL:
	case OpSubgroupAvcRefConvertToMceResultINTEL:
	case OpSubgroupAvcSicInitializeINTEL:
	case OpSubgroupAvcSicConfigureSkcINTEL:
	case OpSubgroupAvcSicConfigureIpeLumaINTEL:
	case OpSubgroupAvcSicConfigureIpeLumaChromaINTEL:
	case OpSubgroupAvcSicGetMotionVectorMaskINTEL:
	case OpSubgroupAvcSicConvertToMcePayloadINTEL:
	case OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL:
	case OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL:
	case OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL:
	case OpSubgroupAvcSicSetBilinearFilterEnableINTEL:
	case OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL:
	case OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL:
	case OpSubgroupAvcSicEvaluateIpeINTEL:
	case OpSubgroupAvcSicEvaluateWithSingleReferenceINTEL:
	case OpSubgroupAvcSicEvaluateWithDualReferenceINTEL:
	case OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL:
	case OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL:
	case OpSubgroupAvcSicConvertToMceResultINTEL:
	case OpSubgroupAvcSicGetIpeLumaShapeINTEL:
	case OpSubgroupAvcSicGetBestIpeLumaDistortionINTEL:
	case OpSubgroupAvcSicGetBestIpeChromaDistortionINTEL:
	case OpSubgroupAvcSicGetPackedIpeLumaModesINTEL:
	case OpSubgroupAvcSicGetIpeChromaModeINTEL:
	case OpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL:
	case OpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL:
	case OpSubgroupAvcSicGetInterRawSadsINTEL:
	case OpVariableLengthArrayINTEL:
	case OpSaveMemoryINTEL:
	case OpRestoreMemoryINTEL:
	case OpArbitraryFloatSinCosPiALTERA:
	case OpArbitraryFloatCastALTERA:
	case OpArbitraryFloatCastFromIntALTERA:
	case OpArbitraryFloatCastToIntALTERA:
	case OpArbitraryFloatAddALTERA:
	case OpArbitraryFloatSubALTERA:
	case OpArbitraryFloatMulALTERA:
	case OpArbitraryFloatDivALTERA:
	case OpArbitraryFloatGTALTERA:
	case OpArbitraryFloatGEALTERA:
	case OpArbitraryFloatLTALTERA:
	case OpArbitraryFloatLEALTERA:
	case OpArbitraryFloatEQALTERA:
	case OpArbitraryFloatRecipALTERA:
	case OpArbitraryFloatRSqrtALTERA:
	case OpArbitraryFloatCbrtALTERA:
	case OpArbitraryFloatHypotALTERA:
	case OpArbitraryFloatSqrtALTERA:
	case OpArbitraryFloatLogINTEL:
	case OpArbitraryFloatLog2INTEL:
	case OpArbitraryFloatLog10INTEL:
	case OpArbitraryFloatLog1pINTEL:
	case OpArbitraryFloatExpINTEL:
	case OpArbitraryFloatExp2INTEL:
	case OpArbitraryFloatExp10INTEL:
	case OpArbitraryFloatExpm1INTEL:
	case OpArbitraryFloatSinINTEL:
	case OpArbitraryFloatCosINTEL:
	case OpArbitraryFloatSinCosINTEL:
	case OpArbitraryFloatSinPiINTEL:
	case OpArbitraryFloatCosPiINTEL:
	case OpArbitraryFloatASinINTEL:
	case OpArbitraryFloatASinPiINTEL:
	case OpArbitraryFloatACosINTEL:
	case OpArbitraryFloatACosPiINTEL:
	case OpArbitraryFloatATanINTEL:
	case OpArbitraryFloatATanPiINTEL:
	case OpArbitraryFloatATan2INTEL:
	case OpArbitraryFloatPowINTEL:
	case OpArbitraryFloatPowRINTEL:
	case OpArbitraryFloatPowNINTEL:
	case OpLoopControlINTEL:
	case OpAliasDomainDeclINTEL:
	case OpAliasScopeDeclINTEL:
	case OpAliasScopeListDeclINTEL:
	case OpFixedSqrtALTERA:
	case OpFixedRecipALTERA:
	case OpFixedRsqrtALTERA:
	case OpFixedSinALTERA:
	case OpFixedCosALTERA:
	case OpFixedSinCosALTERA:
	case OpFixedSinPiALTERA:
	case OpFixedCosPiALTERA:
	case OpFixedSinCosPiALTERA:
	case OpFixedLogALTERA:
	case OpFixedExpALTERA:
	case OpPtrCastToCrossWorkgroupALTERA:
	case OpCrossWorkgroupCastToPtrALTERA:
	case OpReadPipeBlockingALTERA:
	case OpWritePipeBlockingALTERA:
	case OpFPGARegALTERA:
	case OpRayQueryGetRayTMinKHR:
	case OpRayQueryGetRayFlagsKHR:
	case OpRayQueryGetIntersectionTKHR:
	case OpRayQueryGetIntersectionInstanceCustomIndexKHR:
	case OpRayQueryGetIntersectionInstanceIdKHR:
	case OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
	case OpRayQueryGetIntersectionGeometryIndexKHR:
	case OpRayQueryGetIntersectionPrimitiveIndexKHR:
	case OpRayQueryGetIntersectionBarycentricsKHR:
	case OpRayQueryGetIntersectionFrontFaceKHR:
	case OpRayQueryGetIntersectionCandidateAABBOpaqueKHR:
	case OpRayQueryGetIntersectionObjectRayDirectionKHR:
	case OpRayQueryGetIntersectionObjectRayOriginKHR:
	case OpRayQueryGetWorldRayDirectionKHR:
	case OpRayQueryGetWorldRayOriginKHR:
	case OpRayQueryGetIntersectionObjectToWorldKHR:
	case OpRayQueryGetIntersectionWorldToObjectKHR:
	case OpAtomicFAddEXT:
	case OpTypeBufferSurfaceINTEL:
	case OpTypeStructContinuedINTEL:
	case OpConstantCompositeContinuedINTEL:
	case OpSpecConstantCompositeContinuedINTEL:
	case OpCompositeConstructContinuedINTEL:
	case OpConvertFToBF16INTEL:
	case OpConvertBF16ToFINTEL:
	case OpControlBarrierArriveINTEL:
	case OpControlBarrierWaitINTEL:
	case OpArithmeticFenceEXT:
	case OpTaskSequenceCreateALTERA:
	case OpTaskSequenceAsyncALTERA:
	case OpTaskSequenceGetALTERA:
	case OpTaskSequenceReleaseALTERA:
	case OpTypeTaskSequenceALTERA:
	case OpSubgroupBlockPrefetchINTEL:
	case OpSubgroup2DBlockLoadINTEL:
	case OpSubgroup2DBlockLoadTransformINTEL:
	case OpSubgroup2DBlockLoadTransposeINTEL:
	case OpSubgroup2DBlockPrefetchINTEL:
	case OpSubgroup2DBlockStoreINTEL:
	case OpSubgroupMatrixMultiplyAccumulateINTEL:
	case OpBitwiseFunctionINTEL:
	case OpUntypedVariableLengthArrayINTEL:
	case OpConditionalExtensionINTEL:
	case OpConditionalEntryPointINTEL:
	case OpConditionalCapabilityINTEL:
	case OpSpecConstantTargetINTEL:
	case OpSpecConstantArchitectureINTEL:
	case OpSpecConstantCapabilitiesINTEL:
	case OpConditionalCopyObjectINTEL:
	case OpGroupIMulKHR:
	case OpGroupFMulKHR:
	case OpGroupBitwiseAndKHR:
	case OpGroupBitwiseOrKHR:
	case OpGroupBitwiseXorKHR:
	case OpGroupLogicalAndKHR:
	case OpGroupLogicalOrKHR:
	case OpGroupLogicalXorKHR:
	case OpRoundFToTF32INTEL:
	case OpMaskedGatherINTEL:
	case OpMaskedScatterINTEL:
	case OpConvertHandleToImageINTEL:
	case OpConvertHandleToSamplerINTEL:
	case OpConvertHandleToSampledImageINTEL:
	case OpFDot2MixAcc32VALVE:
	case OpFDot2MixAcc16VALVE:
	case OpFDot4MixAcc32VALVE:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid StorageClass enumerator known to this header.
static bool is_valid_storage_class(uint32_t v)
{
	switch (v)
	{
	case StorageClassUniformConstant:
	case StorageClassInput:
	case StorageClassUniform:
	case StorageClassOutput:
	case StorageClassWorkgroup:
	case StorageClassCrossWorkgroup:
	case StorageClassPrivate:
	case StorageClassFunction:
	case StorageClassGeneric:
	case StorageClassPushConstant:
	case StorageClassAtomicCounter:
	case StorageClassImage:
	case StorageClassStorageBuffer:
	case StorageClassTileImageEXT:
	case StorageClassTileAttachmentQCOM:
	case StorageClassNodePayloadAMDX:
	case StorageClassCallableDataKHR:
	case StorageClassIncomingCallableDataKHR:
	case StorageClassRayPayloadKHR:
	case StorageClassHitAttributeKHR:
	case StorageClassIncomingRayPayloadKHR:
	case StorageClassShaderRecordBufferKHR:
	case StorageClassPhysicalStorageBuffer:
	case StorageClassHitObjectAttributeNV:
	case StorageClassTaskPayloadWorkgroupEXT:
	case StorageClassHitObjectAttributeEXT:
	case StorageClassCodeSectionINTEL:
	case StorageClassDeviceOnlyALTERA:
	case StorageClassHostOnlyALTERA:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid Dim enumerator known to this header.
static bool is_valid_dim(uint32_t v)
{
	switch (v)
	{
	case Dim1D:
	case Dim2D:
	case Dim3D:
	case DimCube:
	case DimRect:
	case DimBuffer:
	case DimSubpassData:
	case DimTileImageDataEXT:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid ImageFormat enumerator known to this header.
static bool is_valid_image_format(uint32_t v)
{
	switch (v)
	{
	case ImageFormatUnknown:
	case ImageFormatRgba32f:
	case ImageFormatRgba16f:
	case ImageFormatR32f:
	case ImageFormatRgba8:
	case ImageFormatRgba8Snorm:
	case ImageFormatRg32f:
	case ImageFormatRg16f:
	case ImageFormatR11fG11fB10f:
	case ImageFormatR16f:
	case ImageFormatRgba16:
	case ImageFormatRgb10A2:
	case ImageFormatRg16:
	case ImageFormatRg8:
	case ImageFormatR16:
	case ImageFormatR8:
	case ImageFormatRgba16Snorm:
	case ImageFormatRg16Snorm:
	case ImageFormatRg8Snorm:
	case ImageFormatR16Snorm:
	case ImageFormatR8Snorm:
	case ImageFormatRgba32i:
	case ImageFormatRgba16i:
	case ImageFormatRgba8i:
	case ImageFormatR32i:
	case ImageFormatRg32i:
	case ImageFormatRg16i:
	case ImageFormatRg8i:
	case ImageFormatR16i:
	case ImageFormatR8i:
	case ImageFormatRgba32ui:
	case ImageFormatRgba16ui:
	case ImageFormatRgba8ui:
	case ImageFormatR32ui:
	case ImageFormatRgb10a2ui:
	case ImageFormatRg32ui:
	case ImageFormatRg16ui:
	case ImageFormatRg8ui:
	case ImageFormatR16ui:
	case ImageFormatR8ui:
	case ImageFormatR64ui:
	case ImageFormatR64i:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid AccessQualifier enumerator known to this header.
static bool is_valid_access_qualifier(uint32_t v)
{
	switch (v)
	{
	case AccessQualifierReadOnly:
	case AccessQualifierWriteOnly:
	case AccessQualifierReadWrite:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid AddressingModel enumerator known to this header.
static bool is_valid_addressing_model(uint32_t v)
{
	switch (v)
	{
	case AddressingModelLogical:
	case AddressingModelPhysical32:
	case AddressingModelPhysical64:
	case AddressingModelPhysicalStorageBuffer64:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid MemoryModel enumerator known to this header.
static bool is_valid_memory_model(uint32_t v)
{
	switch (v)
	{
	case MemoryModelSimple:
	case MemoryModelGLSL450:
	case MemoryModelOpenCL:
	case MemoryModelVulkan:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid SourceLanguage enumerator known to this header.
static bool is_valid_source_language(uint32_t v)
{
	switch (v)
	{
	case SourceLanguageUnknown:
	case SourceLanguageESSL:
	case SourceLanguageGLSL:
	case SourceLanguageOpenCL_C:
	case SourceLanguageOpenCL_CPP:
	case SourceLanguageHLSL:
	case SourceLanguageCPP_for_OpenCL:
	case SourceLanguageSYCL:
	case SourceLanguageHERO_C:
	case SourceLanguageNZSL:
	case SourceLanguageWGSL:
	case SourceLanguageSlang:
	case SourceLanguageZig:
	case SourceLanguageRust:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid ExecutionModel enumerator known to this header.
static bool is_valid_execution_model(uint32_t v)
{
	switch (v)
	{
	case ExecutionModelVertex:
	case ExecutionModelTessellationControl:
	case ExecutionModelTessellationEvaluation:
	case ExecutionModelGeometry:
	case ExecutionModelFragment:
	case ExecutionModelGLCompute:
	case ExecutionModelKernel:
	case ExecutionModelTaskNV:
	case ExecutionModelMeshNV:
	case ExecutionModelRayGenerationKHR:
	case ExecutionModelIntersectionKHR:
	case ExecutionModelAnyHitKHR:
	case ExecutionModelClosestHitKHR:
	case ExecutionModelMissKHR:
	case ExecutionModelCallableKHR:
	case ExecutionModelTaskEXT:
	case ExecutionModelMeshEXT:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid ExecutionMode enumerator known to this header.
static bool is_valid_execution_mode(uint32_t v)
{
	switch (v)
	{
	case ExecutionModeInvocations:
	case ExecutionModeSpacingEqual:
	case ExecutionModeSpacingFractionalEven:
	case ExecutionModeSpacingFractionalOdd:
	case ExecutionModeVertexOrderCw:
	case ExecutionModeVertexOrderCcw:
	case ExecutionModePixelCenterInteger:
	case ExecutionModeOriginUpperLeft:
	case ExecutionModeOriginLowerLeft:
	case ExecutionModeEarlyFragmentTests:
	case ExecutionModePointMode:
	case ExecutionModeXfb:
	case ExecutionModeDepthReplacing:
	case ExecutionModeDepthGreater:
	case ExecutionModeDepthLess:
	case ExecutionModeDepthUnchanged:
	case ExecutionModeLocalSize:
	case ExecutionModeLocalSizeHint:
	case ExecutionModeInputPoints:
	case ExecutionModeInputLines:
	case ExecutionModeInputLinesAdjacency:
	case ExecutionModeTriangles:
	case ExecutionModeInputTrianglesAdjacency:
	case ExecutionModeQuads:
	case ExecutionModeIsolines:
	case ExecutionModeOutputVertices:
	case ExecutionModeOutputPoints:
	case ExecutionModeOutputLineStrip:
	case ExecutionModeOutputTriangleStrip:
	case ExecutionModeVecTypeHint:
	case ExecutionModeContractionOff:
	case ExecutionModeInitializer:
	case ExecutionModeFinalizer:
	case ExecutionModeSubgroupSize:
	case ExecutionModeSubgroupsPerWorkgroup:
	case ExecutionModeSubgroupsPerWorkgroupId:
	case ExecutionModeLocalSizeId:
	case ExecutionModeLocalSizeHintId:
	case ExecutionModeNonCoherentColorAttachmentReadEXT:
	case ExecutionModeNonCoherentDepthAttachmentReadEXT:
	case ExecutionModeNonCoherentStencilAttachmentReadEXT:
	case ExecutionModeSubgroupUniformControlFlowKHR:
	case ExecutionModePostDepthCoverage:
	case ExecutionModeDenormPreserve:
	case ExecutionModeDenormFlushToZero:
	case ExecutionModeSignedZeroInfNanPreserve:
	case ExecutionModeRoundingModeRTE:
	case ExecutionModeRoundingModeRTZ:
	case ExecutionModeNonCoherentTileAttachmentReadQCOM:
	case ExecutionModeTileShadingRateQCOM:
	case ExecutionModeEarlyAndLateFragmentTestsAMD:
	case ExecutionModeStencilRefReplacingEXT:
	case ExecutionModeCoalescingAMDX:
	case ExecutionModeIsApiEntryAMDX:
	case ExecutionModeMaxNodeRecursionAMDX:
	case ExecutionModeStaticNumWorkgroupsAMDX:
	case ExecutionModeShaderIndexAMDX:
	case ExecutionModeMaxNumWorkgroupsAMDX:
	case ExecutionModeStencilRefUnchangedFrontAMD:
	case ExecutionModeStencilRefGreaterFrontAMD:
	case ExecutionModeStencilRefLessFrontAMD:
	case ExecutionModeStencilRefUnchangedBackAMD:
	case ExecutionModeStencilRefGreaterBackAMD:
	case ExecutionModeStencilRefLessBackAMD:
	case ExecutionModeQuadDerivativesKHR:
	case ExecutionModeRequireFullQuadsKHR:
	case ExecutionModeSharesInputWithAMDX:
	case ExecutionModeArithmeticPoisonKHR:
	case ExecutionModeOutputLinesEXT:
	case ExecutionModeOutputPrimitivesEXT:
	case ExecutionModeDerivativeGroupQuadsKHR:
	case ExecutionModeDerivativeGroupLinearKHR:
	case ExecutionModeOutputTrianglesEXT:
	case ExecutionModePixelInterlockOrderedEXT:
	case ExecutionModePixelInterlockUnorderedEXT:
	case ExecutionModeSampleInterlockOrderedEXT:
	case ExecutionModeSampleInterlockUnorderedEXT:
	case ExecutionModeShadingRateInterlockOrderedEXT:
	case ExecutionModeShadingRateInterlockUnorderedEXT:
	case ExecutionModeShader64BitIndexingEXT:
	case ExecutionModeSharedLocalMemorySizeINTEL:
	case ExecutionModeRoundingModeRTPINTEL:
	case ExecutionModeRoundingModeRTNINTEL:
	case ExecutionModeFloatingPointModeALTINTEL:
	case ExecutionModeFloatingPointModeIEEEINTEL:
	case ExecutionModeMaxWorkgroupSizeINTEL:
	case ExecutionModeMaxWorkDimINTEL:
	case ExecutionModeNoGlobalOffsetINTEL:
	case ExecutionModeNumSIMDWorkitemsINTEL:
	case ExecutionModeSchedulerTargetFmaxMhzINTEL:
	case ExecutionModeMaximallyReconvergesKHR:
	case ExecutionModeFPFastMathDefault:
	case ExecutionModeStreamingInterfaceINTEL:
	case ExecutionModeRegisterMapInterfaceINTEL:
	case ExecutionModeNamedBarrierCountINTEL:
	case ExecutionModeMaximumRegistersINTEL:
	case ExecutionModeMaximumRegistersIdINTEL:
	case ExecutionModeNamedMaximumRegistersINTEL:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid Decoration enumerator known to this header.
static bool is_valid_decoration(uint32_t v)
{
	switch (v)
	{
	case DecorationRelaxedPrecision:
	case DecorationSpecId:
	case DecorationBlock:
	case DecorationBufferBlock:
	case DecorationRowMajor:
	case DecorationColMajor:
	case DecorationArrayStride:
	case DecorationMatrixStride:
	case DecorationGLSLShared:
	case DecorationGLSLPacked:
	case DecorationCPacked:
	case DecorationBuiltIn:
	case DecorationNoPerspective:
	case DecorationFlat:
	case DecorationPatch:
	case DecorationCentroid:
	case DecorationSample:
	case DecorationInvariant:
	case DecorationRestrict:
	case DecorationAliased:
	case DecorationVolatile:
	case DecorationConstant:
	case DecorationCoherent:
	case DecorationNonWritable:
	case DecorationNonReadable:
	case DecorationUniform:
	case DecorationUniformId:
	case DecorationSaturatedConversion:
	case DecorationStream:
	case DecorationLocation:
	case DecorationComponent:
	case DecorationIndex:
	case DecorationBinding:
	case DecorationDescriptorSet:
	case DecorationOffset:
	case DecorationXfbBuffer:
	case DecorationXfbStride:
	case DecorationFuncParamAttr:
	case DecorationFPRoundingMode:
	case DecorationFPFastMathMode:
	case DecorationLinkageAttributes:
	case DecorationNoContraction:
	case DecorationInputAttachmentIndex:
	case DecorationAlignment:
	case DecorationMaxByteOffset:
	case DecorationAlignmentId:
	case DecorationMaxByteOffsetId:
	case DecorationSaturatedToLargestFloat8NormalConversionEXT:
	case DecorationNoSignedWrap:
	case DecorationNoUnsignedWrap:
	case DecorationWeightTextureQCOM:
	case DecorationBlockMatchTextureQCOM:
	case DecorationBlockMatchSamplerQCOM:
	case DecorationExplicitInterpAMD:
	case DecorationNodeSharesPayloadLimitsWithAMDX:
	case DecorationNodeMaxPayloadsAMDX:
	case DecorationTrackFinishWritingAMDX:
	case DecorationPayloadNodeNameAMDX:
	case DecorationPayloadNodeBaseIndexAMDX:
	case DecorationPayloadNodeSparseArrayAMDX:
	case DecorationPayloadNodeArraySizeAMDX:
	case DecorationPayloadDispatchIndirectAMDX:
	case DecorationArrayStrideIdEXT:
	case DecorationOffsetIdEXT:
	case DecorationUTFEncodedKHR:
	case DecorationOverrideCoverageNV:
	case DecorationPassthroughNV:
	case DecorationViewportRelativeNV:
	case DecorationSecondaryViewportRelativeNV:
	case DecorationPerPrimitiveEXT:
	case DecorationPerViewNV:
	case DecorationPerTaskNV:
	case DecorationPerVertexKHR:
	case DecorationNonUniform:
	case DecorationRestrictPointer:
	case DecorationAliasedPointer:
	case DecorationMemberOffsetNV:
	case DecorationHitObjectShaderRecordBufferNV:
	case DecorationHitObjectShaderRecordBufferEXT:
	case DecorationBankNV:
	case DecorationBindlessSamplerNV:
	case DecorationBindlessImageNV:
	case DecorationBoundSamplerNV:
	case DecorationBoundImageNV:
	case DecorationSIMTCallINTEL:
	case DecorationReferencedIndirectlyINTEL:
	case DecorationClobberINTEL:
	case DecorationSideEffectsINTEL:
	case DecorationVectorComputeVariableINTEL:
	case DecorationFuncParamIOKindINTEL:
	case DecorationVectorComputeFunctionINTEL:
	case DecorationStackCallINTEL:
	case DecorationGlobalVariableOffsetINTEL:
	case DecorationCounterBuffer:
	case DecorationHlslSemanticGOOGLE:
	case DecorationUserTypeGOOGLE:
	case DecorationFunctionRoundingModeINTEL:
	case DecorationFunctionDenormModeINTEL:
	case DecorationRegisterALTERA:
	case DecorationMemoryALTERA:
	case DecorationNumbanksALTERA:
	case DecorationBankwidthALTERA:
	case DecorationMaxPrivateCopiesALTERA:
	case DecorationSinglepumpALTERA:
	case DecorationDoublepumpALTERA:
	case DecorationMaxReplicatesALTERA:
	case DecorationSimpleDualPortALTERA:
	case DecorationMergeALTERA:
	case DecorationBankBitsALTERA:
	case DecorationForcePow2DepthALTERA:
	case DecorationStridesizeALTERA:
	case DecorationWordsizeALTERA:
	case DecorationTrueDualPortALTERA:
	case DecorationBurstCoalesceALTERA:
	case DecorationCacheSizeALTERA:
	case DecorationDontStaticallyCoalesceALTERA:
	case DecorationPrefetchALTERA:
	case DecorationStallEnableALTERA:
	case DecorationFuseLoopsInFunctionALTERA:
	case DecorationMathOpDSPModeALTERA:
	case DecorationAliasScopeINTEL:
	case DecorationNoAliasINTEL:
	case DecorationInitiationIntervalALTERA:
	case DecorationMaxConcurrencyALTERA:
	case DecorationPipelineEnableALTERA:
	case DecorationBufferLocationALTERA:
	case DecorationIOPipeStorageALTERA:
	case DecorationFunctionFloatingPointModeINTEL:
	case DecorationSingleElementVectorINTEL:
	case DecorationVectorComputeCallableFunctionINTEL:
	case DecorationMediaBlockIOINTEL:
	case DecorationStallFreeALTERA:
	case DecorationFPMaxErrorDecorationINTEL:
	case DecorationLatencyControlLabelALTERA:
	case DecorationLatencyControlConstraintALTERA:
	case DecorationConduitKernelArgumentALTERA:
	case DecorationRegisterMapKernelArgumentALTERA:
	case DecorationMMHostInterfaceAddressWidthALTERA:
	case DecorationMMHostInterfaceDataWidthALTERA:
	case DecorationMMHostInterfaceLatencyALTERA:
	case DecorationMMHostInterfaceReadWriteModeALTERA:
	case DecorationMMHostInterfaceMaxBurstALTERA:
	case DecorationMMHostInterfaceWaitRequestALTERA:
	case DecorationStableKernelArgumentALTERA:
	case DecorationHostAccessINTEL:
	case DecorationInitModeALTERA:
	case DecorationImplementInRegisterMapALTERA:
	case DecorationConditionalINTEL:
	case DecorationCacheControlLoadINTEL:
	case DecorationCacheControlStoreINTEL:
		return true;
	default:
		return false;
	}
}

// Returns true if the literal is a valid Capability enumerator known to this header.
static bool is_valid_capability(uint32_t v)
{
	switch (v)
	{
	case CapabilityMatrix:
	case CapabilityShader:
	case CapabilityGeometry:
	case CapabilityTessellation:
	case CapabilityAddresses:
	case CapabilityLinkage:
	case CapabilityKernel:
	case CapabilityVector16:
	case CapabilityFloat16Buffer:
	case CapabilityFloat16:
	case CapabilityFloat64:
	case CapabilityInt64:
	case CapabilityInt64Atomics:
	case CapabilityImageBasic:
	case CapabilityImageReadWrite:
	case CapabilityImageMipmap:
	case CapabilityPipes:
	case CapabilityGroups:
	case CapabilityDeviceEnqueue:
	case CapabilityLiteralSampler:
	case CapabilityAtomicStorage:
	case CapabilityInt16:
	case CapabilityTessellationPointSize:
	case CapabilityGeometryPointSize:
	case CapabilityImageGatherExtended:
	case CapabilityStorageImageMultisample:
	case CapabilityUniformBufferArrayDynamicIndexing:
	case CapabilitySampledImageArrayDynamicIndexing:
	case CapabilityStorageBufferArrayDynamicIndexing:
	case CapabilityStorageImageArrayDynamicIndexing:
	case CapabilityClipDistance:
	case CapabilityCullDistance:
	case CapabilityImageCubeArray:
	case CapabilitySampleRateShading:
	case CapabilityImageRect:
	case CapabilitySampledRect:
	case CapabilityGenericPointer:
	case CapabilityInt8:
	case CapabilityInputAttachment:
	case CapabilitySparseResidency:
	case CapabilityMinLod:
	case CapabilitySampled1D:
	case CapabilityImage1D:
	case CapabilitySampledCubeArray:
	case CapabilitySampledBuffer:
	case CapabilityImageBuffer:
	case CapabilityImageMSArray:
	case CapabilityStorageImageExtendedFormats:
	case CapabilityImageQuery:
	case CapabilityDerivativeControl:
	case CapabilityInterpolationFunction:
	case CapabilityTransformFeedback:
	case CapabilityGeometryStreams:
	case CapabilityStorageImageReadWithoutFormat:
	case CapabilityStorageImageWriteWithoutFormat:
	case CapabilityMultiViewport:
	case CapabilitySubgroupDispatch:
	case CapabilityNamedBarrier:
	case CapabilityPipeStorage:
	case CapabilityGroupNonUniform:
	case CapabilityGroupNonUniformVote:
	case CapabilityGroupNonUniformArithmetic:
	case CapabilityGroupNonUniformBallot:
	case CapabilityGroupNonUniformShuffle:
	case CapabilityGroupNonUniformShuffleRelative:
	case CapabilityGroupNonUniformClustered:
	case CapabilityGroupNonUniformQuad:
	case CapabilityShaderLayer:
	case CapabilityShaderViewportIndex:
	case CapabilityUniformDecoration:
	case CapabilityCoreBuiltinsARM:
	case CapabilityTileImageColorReadAccessEXT:
	case CapabilityTileImageDepthReadAccessEXT:
	case CapabilityTileImageStencilReadAccessEXT:
	case CapabilityTensorsARM:
	case CapabilityStorageTensorArrayDynamicIndexingARM:
	case CapabilityStorageTensorArrayNonUniformIndexingARM:
	case CapabilityGraphARM:
	case CapabilityCooperativeMatrixLayoutsARM:
	case CapabilityFloat8EXT:
	case CapabilityFloat8CooperativeMatrixEXT:
	case CapabilityFragmentShadingRateKHR:
	case CapabilitySubgroupBallotKHR:
	case CapabilityDrawParameters:
	case CapabilityWorkgroupMemoryExplicitLayoutKHR:
	case CapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR:
	case CapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR:
	case CapabilitySubgroupVoteKHR:
	case CapabilityStorageBuffer16BitAccess:
	case CapabilityStorageUniform16:
	case CapabilityStoragePushConstant16:
	case CapabilityStorageInputOutput16:
	case CapabilityDeviceGroup:
	case CapabilityMultiView:
	case CapabilityVariablePointersStorageBuffer:
	case CapabilityVariablePointers:
	case CapabilityAtomicStorageOps:
	case CapabilitySampleMaskPostDepthCoverage:
	case CapabilityStorageBuffer8BitAccess:
	case CapabilityUniformAndStorageBuffer8BitAccess:
	case CapabilityStoragePushConstant8:
	case CapabilityDenormPreserve:
	case CapabilityDenormFlushToZero:
	case CapabilitySignedZeroInfNanPreserve:
	case CapabilityRoundingModeRTE:
	case CapabilityRoundingModeRTZ:
	case CapabilityRayQueryProvisionalKHR:
	case CapabilityRayQueryKHR:
	case CapabilityUntypedPointersKHR:
	case CapabilityRayTraversalPrimitiveCullingKHR:
	case CapabilityRayTracingKHR:
	case CapabilityTextureSampleWeightedQCOM:
	case CapabilityTextureBoxFilterQCOM:
	case CapabilityTextureBlockMatchQCOM:
	case CapabilityTileShadingQCOM:
	case CapabilityCooperativeMatrixConversionQCOM:
	case CapabilityTextureBlockMatch2QCOM:
	case CapabilityFloat16ImageAMD:
	case CapabilityImageGatherBiasLodAMD:
	case CapabilityFragmentMaskAMD:
	case CapabilityStencilExportEXT:
	case CapabilityImageReadWriteLodAMD:
	case CapabilityInt64ImageEXT:
	case CapabilityShaderClockKHR:
	case CapabilityShaderEnqueueAMDX:
	case CapabilityQuadControlKHR:
	case CapabilityInt4TypeINTEL:
	case CapabilityInt4CooperativeMatrixINTEL:
	case CapabilityBFloat16TypeKHR:
	case CapabilityBFloat16DotProductKHR:
	case CapabilityBFloat16CooperativeMatrixKHR:
	case CapabilityAbortKHR:
	case CapabilityDescriptorHeapEXT:
	case CapabilityConstantDataKHR:
	case CapabilityPoisonFreezeKHR:
	case CapabilitySampleMaskOverrideCoverageNV:
	case CapabilityGeometryShaderPassthroughNV:
	case CapabilityShaderViewportIndexLayerEXT:
	case CapabilityShaderViewportMaskNV:
	case CapabilityShaderStereoViewNV:
	case CapabilityPerViewAttributesNV:
	case CapabilityFragmentFullyCoveredEXT:
	case CapabilityMeshShadingNV:
	case CapabilityImageFootprintNV:
	case CapabilityMeshShadingEXT:
	case CapabilityFragmentBarycentricKHR:
	case CapabilityComputeDerivativeGroupQuadsKHR:
	case CapabilityFragmentDensityEXT:
	case CapabilityGroupNonUniformPartitionedEXT:
	case CapabilityShaderNonUniform:
	case CapabilityRuntimeDescriptorArray:
	case CapabilityInputAttachmentArrayDynamicIndexing:
	case CapabilityUniformTexelBufferArrayDynamicIndexing:
	case CapabilityStorageTexelBufferArrayDynamicIndexing:
	case CapabilityUniformBufferArrayNonUniformIndexing:
	case CapabilitySampledImageArrayNonUniformIndexing:
	case CapabilityStorageBufferArrayNonUniformIndexing:
	case CapabilityStorageImageArrayNonUniformIndexing:
	case CapabilityInputAttachmentArrayNonUniformIndexing:
	case CapabilityUniformTexelBufferArrayNonUniformIndexing:
	case CapabilityStorageTexelBufferArrayNonUniformIndexing:
	case CapabilityRayTracingPositionFetchKHR:
	case CapabilityRayTracingNV:
	case CapabilityRayTracingMotionBlurNV:
	case CapabilityVulkanMemoryModel:
	case CapabilityVulkanMemoryModelDeviceScope:
	case CapabilityPhysicalStorageBufferAddresses:
	case CapabilityComputeDerivativeGroupLinearKHR:
	case CapabilityRayTracingProvisionalKHR:
	case CapabilityCooperativeMatrixNV:
	case CapabilityFragmentShaderSampleInterlockEXT:
	case CapabilityFragmentShaderShadingRateInterlockEXT:
	case CapabilityShaderSMBuiltinsNV:
	case CapabilityFragmentShaderPixelInterlockEXT:
	case CapabilityDemoteToHelperInvocation:
	case CapabilityDisplacementMicromapNV:
	case CapabilityRayTracingOpacityMicromapEXT:
	case CapabilityShaderInvocationReorderNV:
	case CapabilityShaderInvocationReorderEXT:
	case CapabilityBindlessTextureNV:
	case CapabilityRayQueryPositionFetchKHR:
	case CapabilityCooperativeVectorNV:
	case CapabilityAtomicFloat16VectorNV:
	case CapabilityRayTracingDisplacementMicromapNV:
	case CapabilityRawAccessChainsNV:
	case CapabilityRayTracingSpheresGeometryNV:
	case CapabilityRayTracingLinearSweptSpheresGeometryNV:
	case CapabilityPushConstantBanksNV:
	case CapabilityLongVectorEXT:
	case CapabilityShader64BitIndexingEXT:
	case CapabilityCooperativeMatrixReductionsNV:
	case CapabilityCooperativeMatrixConversionsNV:
	case CapabilityCooperativeMatrixPerElementOperationsNV:
	case CapabilityCooperativeMatrixTensorAddressingNV:
	case CapabilityCooperativeMatrixBlockLoadsNV:
	case CapabilityCooperativeVectorTrainingNV:
	case CapabilityRayTracingClusterAccelerationStructureNV:
	case CapabilityTensorAddressingNV:
	case CapabilitySubgroupShuffleINTEL:
	case CapabilitySubgroupBufferBlockIOINTEL:
	case CapabilitySubgroupImageBlockIOINTEL:
	case CapabilitySubgroupImageMediaBlockIOINTEL:
	case CapabilityRoundToInfinityINTEL:
	case CapabilityFloatingPointModeINTEL:
	case CapabilityIntegerFunctions2INTEL:
	case CapabilityFunctionPointersINTEL:
	case CapabilityIndirectReferencesINTEL:
	case CapabilityAsmINTEL:
	case CapabilityAtomicFloat32MinMaxEXT:
	case CapabilityAtomicFloat64MinMaxEXT:
	case CapabilityAtomicFloat16MinMaxEXT:
	case CapabilityVectorComputeINTEL:
	case CapabilityVectorAnyINTEL:
	case CapabilityExpectAssumeKHR:
	case CapabilitySubgroupAvcMotionEstimationINTEL:
	case CapabilitySubgroupAvcMotionEstimationIntraINTEL:
	case CapabilitySubgroupAvcMotionEstimationChromaINTEL:
	case CapabilityVariableLengthArrayINTEL:
	case CapabilityFunctionFloatControlINTEL:
	case CapabilityFPGAMemoryAttributesALTERA:
	case CapabilityFPFastMathModeINTEL:
	case CapabilityArbitraryPrecisionIntegersALTERA:
	case CapabilityArbitraryPrecisionFloatingPointALTERA:
	case CapabilityUnstructuredLoopControlsINTEL:
	case CapabilityFPGALoopControlsALTERA:
	case CapabilityKernelAttributesINTEL:
	case CapabilityFPGAKernelAttributesINTEL:
	case CapabilityFPGAMemoryAccessesALTERA:
	case CapabilityFPGAClusterAttributesALTERA:
	case CapabilityLoopFuseALTERA:
	case CapabilityFPGADSPControlALTERA:
	case CapabilityMemoryAccessAliasingINTEL:
	case CapabilityFPGAInvocationPipeliningAttributesALTERA:
	case CapabilityFPGABufferLocationALTERA:
	case CapabilityArbitraryPrecisionFixedPointALTERA:
	case CapabilityUSMStorageClassesALTERA:
	case CapabilityRuntimeAlignedAttributeALTERA:
	case CapabilityIOPipesALTERA:
	case CapabilityBlockingPipesALTERA:
	case CapabilityFPGARegALTERA:
	case CapabilityDotProductInputAll:
	case CapabilityDotProductInput4x8Bit:
	case CapabilityDotProductInput4x8BitPacked:
	case CapabilityDotProduct:
	case CapabilityRayCullMaskKHR:
	case CapabilityCooperativeMatrixKHR:
	case CapabilityReplicatedCompositesEXT:
	case CapabilityBitInstructions:
	case CapabilityGroupNonUniformRotateKHR:
	case CapabilityFloatControls2:
	case CapabilityFMAKHR:
	case CapabilityAtomicFloat32AddEXT:
	case CapabilityAtomicFloat64AddEXT:
	case CapabilityLongCompositesINTEL:
	case CapabilityOptNoneEXT:
	case CapabilityAtomicFloat16AddEXT:
	case CapabilityDebugInfoModuleINTEL:
	case CapabilityBFloat16ConversionINTEL:
	case CapabilitySplitBarrierINTEL:
	case CapabilityArithmeticFenceEXT:
	case CapabilityFPGAClusterAttributesV2ALTERA:
	case CapabilityFPGAKernelAttributesv2INTEL:
	case CapabilityTaskSequenceALTERA:
	case CapabilityFPMaxErrorINTEL:
	case CapabilityFPGALatencyControlALTERA:
	case CapabilityFPGAArgumentInterfacesALTERA:
	case CapabilityGlobalVariableHostAccessINTEL:
	case CapabilityGlobalVariableFPGADecorationsALTERA:
	case CapabilitySubgroupBufferPrefetchINTEL:
	case CapabilitySubgroup2DBlockIOINTEL:
	case CapabilitySubgroup2DBlockTransformINTEL:
	case CapabilitySubgroup2DBlockTransposeINTEL:
	case CapabilitySubgroupMatrixMultiplyAccumulateINTEL:
	case CapabilityTernaryBitwiseFunctionINTEL:
	case CapabilityUntypedVariableLengthArrayINTEL:
	case CapabilitySpecConditionalINTEL:
	case CapabilityFunctionVariantsINTEL:
	case CapabilityGroupUniformArithmeticKHR:
	case CapabilityTensorFloat32RoundingINTEL:
	case CapabilityMaskedGatherScatterINTEL:
	case CapabilityCacheControlsINTEL:
	case CapabilityRegisterLimitsINTEL:
	case CapabilityBindlessImagesINTEL:
	case CapabilityDotProductFloat16AccFloat32VALVE:
	case CapabilityDotProductFloat16AccFloat16VALVE:
	case CapabilityDotProductBFloat16AccVALVE:
	case CapabilityDotProductFloat8AccFloat32VALVE:
		return true;
	default:
		return false;
	}
}


void Parser::parse()
{
	auto &spirv = ir.spirv;

	auto len = spirv.size();
	if (len < 5)
		SPIRV_CROSS_THROW("SPIRV file too small.");

	auto s = spirv.data();

	// Endian-swap if we need to.
	if (s[0] == swap_endian(MagicNumber))
		transform(begin(spirv), end(spirv), begin(spirv), [](uint32_t c) { return swap_endian(c); });

	if (s[0] != MagicNumber || !is_valid_spirv_version(s[1]))
		SPIRV_CROSS_THROW("Invalid SPIRV format.");

	uint32_t bound = s[3];

	const uint32_t MaximumNumberOfIDs = 0x3fffff;
	if (bound > MaximumNumberOfIDs)
		SPIRV_CROSS_THROW("ID bound exceeds limit of 0x3fffff.\n");

	ir.set_id_bounds(bound);

	uint32_t offset = 5;

	SmallVector<Instruction> instructions;
	while (offset < len)
	{
		Instruction instr = {};
		instr.op = spirv[offset] & 0xffff;
		instr.count = (spirv[offset] >> 16) & 0xffff;

		if (instr.count == 0)
			SPIRV_CROSS_THROW("SPIR-V instructions cannot consume 0 words. Invalid SPIR-V file. (at instruction offset " +
			                  std::to_string(offset) + ")");

		instr.offset = offset + 1;
		instr.length = instr.count - 1;

		offset += instr.count;

		if (offset > spirv.size())
			SPIRV_CROSS_THROW("SPIR-V instruction goes out of bounds. (at instruction offset " +
			                  std::to_string(instr.offset - 1) + ")");

		instructions.push_back(instr);
	}

#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	for (auto &i : instructions)
	{
		try
		{
			parse(i);
		}
		catch (CompilerError &e)
		{
			// Annotate the failure with the word offset of the offending
			// instruction so malformed modules can be diagnosed.
			SPIRV_CROSS_THROW(std::string(e.what()) + " (at instruction offset " +
			                  std::to_string(i.offset - 1) + ")");
		}
	}
#else
	for (auto &i : instructions)
		parse(i);
#endif

	for (auto &fixup : forward_pointer_fixups)
	{
		auto &target = get<SPIRType>(fixup.first);
		auto &source = get<SPIRType>(fixup.second);
		target.member_types = source.member_types;
		target.basetype = source.basetype;
		target.self = source.self;
	}
	forward_pointer_fixups.clear();

	// Reject cyclic value-type containment. Valid modules can only contain
	// cycles through pointers (e.g. linked lists via OpTypeForwardPointer);
	// a cycle through value members would make recursive type layout walks
	// in the compiler overflow the stack. Member IDs which do not name types
	// are rejected here as well via the checked get<>().
	{
		// 0 = unvisited, 1 = on DFS stack, 2 = done.
		SmallVector<uint8_t> visit_state;
		visit_state.resize(ir.ids.size());
		for (uint32_t start = 0; start < ir.ids.size(); start++)
		{
			if (visit_state[start] != 0 || ir.ids[start].get_type() != TypeType)
				continue;

			// Iterative DFS; a recursive walk could itself overflow the
			// stack on very deep (but valid) type graphs.
			SmallVector<std::pair<uint32_t, uint32_t>> dfs_stack;
			dfs_stack.push_back({ start, 0 });
			visit_state[start] = 1;
			while (!dfs_stack.empty())
			{
				auto &frame = dfs_stack.back();
				auto &type = get<SPIRType>(frame.first);

				// Determine the next value-contained child of this type.
				uint32_t child = 0;
				bool has_child = false;
				if (!type.pointer)
				{
					if (!type.array.empty() && frame.second == 0)
					{
						child = type.parent_type;
						has_child = true;
						frame.second = 1;
					}
					else if (type.basetype == SPIRType::Struct && type.array.empty() &&
					         frame.second < type.member_types.size())
					{
						child = type.member_types[frame.second++];
						has_child = true;
					}
				}

				if (!has_child)
				{
					visit_state[frame.first] = 2;
					dfs_stack.pop_back();
					continue;
				}

				auto &child_type = get<SPIRType>(child);
				// Cycles through pointers are legal.
				if (child_type.pointer)
					continue;
				if (visit_state[child] == 1)
					SPIRV_CROSS_THROW("Cyclic value-type containment is not allowed.");
				if (visit_state[child] == 0)
				{
					visit_state[child] = 1;
					dfs_stack.push_back({ child, 0 });
				}
			}
		}
	}

	for (auto &source : ir.sources)
	{
		auto cmp = [](const ParsedIR::Source::Marker &a, const ParsedIR::Source::Marker &b) {
			return a.line < b.line;
		};

		std::sort(source.line_markers.begin(), source.line_markers.end(), cmp);
	}

	if (current_function)
		SPIRV_CROSS_THROW("Function was not terminated.");
	if (current_block)
		SPIRV_CROSS_THROW("Block was not terminated.");

	// Now that all definitions are bound to a kind, we can filter the library
	// exports and populate the exported functions.
	for (uint32_t id : ir.library_exports)
	{
		if (ir.ids[id].get_type() == TypeFunction)
			ir.library_exported_functions.push_back(id);
	}

	if (ir.default_entry_point == 0)
	{
		if (ir.library_exported_functions.empty())
			SPIRV_CROSS_THROW("There is no entry point in the SPIR-V module.");

		// No OpEntryPoint, but the module exports functions. Treat as a library
		// module: designate the first exported function as the default entry
		// point so analyses keyed on default_entry_point can run.
		ir.is_library_module = true;
		ir.default_entry_point = ir.library_exported_functions.front();
		auto &name = ir.get_name(ir.default_entry_point);
		ir.entry_points.insert(std::make_pair(ir.default_entry_point,
		                                      SPIREntryPoint(ir.default_entry_point, ExecutionModelGLCompute, name)));
	}
}

const uint32_t *Parser::stream(const Instruction &instr) const
{
	// If we're not going to use any arguments, just return nullptr.
	// We want to avoid case where we return an out of range pointer
	// that trips debug assertions on some platforms.
	if (!instr.length)
		return nullptr;

	if (instr.offset + instr.length > ir.spirv.size())
		SPIRV_CROSS_THROW("Compiler::stream() out of range.");
	return &ir.spirv[instr.offset];
}

static string extract_string(const std::vector<uint32_t> &spirv, uint32_t offset)
{
	return extract_string(spirv.data() + offset, spirv.size() - offset);
}

// Returns the minimum number of operand words (i.e. excluding the first
// instruction word) that the parser dereferences unconditionally for an
// explicitly handled opcode. Opcodes which are not explicitly handled by the
// parser return 0; their operands are never read here.
static uint32_t min_operand_count(uint32_t op)
{
	switch (op)
	{
	case OpString:
	case OpMemoryModel:
	case OpSource:
	case OpUndef:
	case OpName:
	case OpDecorate:
	case OpDecorateId:
	case OpTypeFloat:
	case OpTypeRuntimeArray:
	case OpTypeSampledImage:
	case OpTypeUntypedPointerKHR:
	case OpTypeForwardPointer:
	case OpTypeFunction:
	case OpTypeTensorARM:
	case OpTypeBufferEXT:
	case OpConstantFalse:
	case OpConstantTrue:
	case OpSpecConstantFalse:
	case OpSpecConstantTrue:
	case OpConstantNull:
	case OpConstantComposite:
	case OpSpecConstantComposite:
	case OpConstantDataKHR:
	case OpSpecConstantDataKHR:
	case OpPhi:
	case OpFunctionParameter:
	case OpSwitch:
	case OpLoopMerge:
	case OpLine:
	case OpExecutionMode:
	case OpExecutionModeId:
	case OpExtInstImport:
	case OpAbortKHR:
		return 2;

	case OpCapability:
	case OpExtension:
	case OpSourceContinued:
	case OpTypeVoid:
	case OpTypeBool:
	case OpTypeStruct:
	case OpTypeSampler:
	case OpTypeAccelerationStructureKHR:
	case OpTypeRayQueryKHR:
	case OpLabel:
	case OpBranch:
	case OpReturnValue:
	case OpSelectionMerge:
	case OpGroupDecorate:
	case OpGroupMemberDecorate:
		return 1;

	case OpEntryPoint:
	case OpMemberName:
	case OpMemberDecorate:
	case OpMemberDecorateIdEXT:
	case OpDecorateStringGOOGLE:
	case OpTypeInt:
	case OpTypeVector:
	case OpTypeMatrix:
	case OpTypeArray:
	case OpTypePointer:
	case OpTypeCooperativeVectorNV:
	case OpConstant:
	case OpSpecConstant:
	case OpConstantCompositeReplicateEXT:
	case OpSpecConstantCompositeReplicateEXT:
	case OpConstantSizeOfEXT:
	case OpSpecConstantOp:
	case OpVariable:
	case OpUntypedVariableKHR:
	case OpBranchConditional:
	case OpEmitMeshTasksEXT:
		return 3;

	case OpMemberDecorateStringGOOGLE:
	case OpFunction:
		return 4;

	case OpTypeCooperativeMatrixKHR:
		return 6;

	case OpTypeImage:
		return 8;

	default:
		return 0;
	}
}

void Parser::parse(const Instruction &instruction)
{
	auto *ops = stream(instruction);
	auto op = static_cast<Op>(instruction.op);
	uint32_t length = instruction.length;

	// Reject opcodes this header does not know about. Unknown opcodes cannot
	// be compiled anyway, and letting them through would store invalid enum
	// values in the IR (undefined behavior on later loads).
	if (!is_known_opcode(instruction.op))
		SPIRV_CROSS_THROW("Unknown opcode " + std::to_string(instruction.op) + " in SPIR-V module.");

	// Reject truncated instructions up front so that operand reads below
	// can never go out of bounds.
	if (length < min_operand_count(instruction.op))
		SPIRV_CROSS_THROW("SPIR-V instruction has too few operands. Invalid SPIR-V file.");

	// Block IDs come straight from the module. block_meta is sized by the
	// declared ID bound, so validate before indexing into it.
	auto block_meta_flags = [this](uint32_t id) -> ParsedIR::BlockMetaFlags & {
		if (id >= ir.block_meta.size())
			SPIRV_CROSS_THROW("Block ID " + std::to_string(id) +
			                  " is out of range for the declared ID bound " +
			                  std::to_string(ir.block_meta.size()) + ".");
		return ir.block_meta[id];
	};

	// HACK for glslang that might emit OpEmitMeshTasksEXT followed by return / branch.
	// Instead of failing hard, just ignore it.
	if (ignore_trailing_block_opcodes)
	{
		ignore_trailing_block_opcodes = false;
		if (op == OpReturn || op == OpBranch || op == OpUnreachable)
			return;
	}

	switch (op)
	{
	case OpSourceExtension:
	case OpNop:
	case OpModuleProcessed:
		break;

	case OpString:
	{
		set<SPIRString>(ops[0], extract_string(ir.spirv, instruction.offset + 1));
		break;
	}

	case OpMemoryModel:
		if (!is_valid_addressing_model(ops[0]) || !is_valid_memory_model(ops[1]))
			SPIRV_CROSS_THROW("Invalid memory model in SPIR-V module.");
		ir.addressing_model = static_cast<AddressingModel>(ops[0]);
		ir.memory_model = static_cast<MemoryModel>(ops[1]);
		break;

	case OpSource:
	{
		if (!is_valid_source_language(ops[0]))
			SPIRV_CROSS_THROW("Invalid source language in SPIR-V module.");
		ir.sources.emplace_back();
		auto &source = ir.sources.back();
		source.lang = static_cast<SourceLanguage>(ops[0]);

		switch (source.lang)
		{
		case SourceLanguageESSL:
			source.es = true;
			source.version = ops[1];
			source.known = true;
			source.hlsl = false;
			break;

		case SourceLanguageGLSL:
			source.es = false;
			source.version = ops[1];
			source.known = true;
			source.hlsl = false;
			break;

		case SourceLanguageHLSL:
			// For purposes of cross-compiling, this is GLSL 450.
			source.es = false;
			source.version = 450;
			source.known = true;
			source.hlsl = true;
			break;

		default:
			source.known = false;
			break;
		}

		if (length >= 3)
			source.file_id = ops[2];

		if (length >= 4)
			source.source = extract_string(ir.spirv, instruction.offset + 3);

		break;
	}

	case OpSourceContinued:
		if (!ir.sources.empty())
			ir.sources.back().source += extract_string(ir.spirv, instruction.offset);
		break;

	case OpUndef:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		set<SPIRUndef>(id, result_type);
		if (current_block)
			current_block->ops.push_back(instruction);
		break;
	}

	case OpCapability:
	{
		uint32_t cap = ops[0];
		if (!is_valid_capability(cap))
			SPIRV_CROSS_THROW("Invalid capability in SPIR-V module.");
		if (cap == CapabilityKernel)
			SPIRV_CROSS_THROW("Kernel capability not supported.");

		ir.declared_capabilities.push_back(static_cast<Capability>(ops[0]));
		break;
	}

	case OpExtension:
	{
		auto ext = extract_string(ir.spirv, instruction.offset);
		ir.declared_extensions.push_back(std::move(ext));
		break;
	}

	case OpExtInstImport:
	{
		uint32_t id = ops[0];

		SPIRExtension::Extension spirv_ext = SPIRExtension::Unsupported;

		auto ext = extract_string(ir.spirv, instruction.offset + 1);
		if (ext == "GLSL.std.450")
			spirv_ext = SPIRExtension::GLSL;
		else if (ext == "DebugInfo")
			spirv_ext = SPIRExtension::SPV_debug_info;
		else if (ext == "SPV_AMD_shader_ballot")
			spirv_ext = SPIRExtension::SPV_AMD_shader_ballot;
		else if (ext == "SPV_AMD_shader_explicit_vertex_parameter")
			spirv_ext = SPIRExtension::SPV_AMD_shader_explicit_vertex_parameter;
		else if (ext == "SPV_AMD_shader_trinary_minmax")
			spirv_ext = SPIRExtension::SPV_AMD_shader_trinary_minmax;
		else if (ext == "SPV_AMD_gcn_shader")
			spirv_ext = SPIRExtension::SPV_AMD_gcn_shader;
		else if (ext == "NonSemantic.DebugPrintf")
			spirv_ext = SPIRExtension::NonSemanticDebugPrintf;
		else if (ext.find("NonSemantic.Shader.DebugInfo.") == 0)
			spirv_ext = SPIRExtension::NonSemanticShaderDebugInfo;
		else if (ext.find("NonSemantic.") == 0)
			spirv_ext = SPIRExtension::NonSemanticGeneric;

		set<SPIRExtension>(id, spirv_ext);
		// Other SPIR-V extensions which have ExtInstrs are currently not supported.

		break;
	}

	case OpExtInst:
	case OpExtInstWithForwardRefsKHR:
	{
		// The SPIR-V debug information extended instructions might come at global scope.
		if (current_block)
		{
			current_block->ops.push_back(instruction);
			if (length >= 2)
			{
				const auto *type = maybe_get<SPIRType>(ops[0]);
				if (type)
					ir.load_type_width.insert({ ops[1], type->width });
			}
		}

		if (op == OpExtInst && length > 4)
		{
			// Don't want to deal with ForwardRefs here.
			auto &ext = get<SPIRExtension>(ops[2]);
			if (ext.ext == SPIRExtension::NonSemanticShaderDebugInfo)
			{
				const auto instr = ops[3];
				if (instr == NonSemanticShaderDebugInfo100DebugSource)
				{
					set<SPIRString>(ops[1], get<SPIRString>(ops[4]).str);

					ir.sources.emplace_back();
					auto &source = ir.sources.back();
					source.file_id = ops[4];
					source.define_id = ops[1];
					if (length >= 6)
						source.source = ir.get<SPIRString>(ops[5]).str;
				}
				else if (instr == NonSemanticShaderDebugInfo100DebugSourceContinued)
				{
					if (length < 5)
						SPIRV_CROSS_THROW("Invalid arguments for ShaderDebugInfo100DebugSourceContinued");
					if (!ir.sources.empty())
						ir.sources.back().source += ir.get<SPIRString>(ops[4]).str;
				}
				else if (instr == NonSemanticShaderDebugInfo100DebugLine)
				{
					if (length < 9)
						SPIRV_CROSS_THROW("Invalid arguments for ShaderDebugInfo100DebugLine");
					auto source_id = ops[4];
					auto line_start = ir.get<SPIRConstant>(ops[5]).scalar_i32();
					auto col_start = ir.get<SPIRConstant>(ops[7]).scalar_i32();

					for (auto &source : ir.sources)
					{
						if (source.define_id != source_id)
							continue;

						source.line_markers.emplace_back();
						auto &marker = source.line_markers.back();
						marker.line = line_start;
						marker.col = col_start;
						marker.offset = instruction.offset - 1;
						marker.function_id = current_function ? current_function->self : ID(0);
						marker.block_id = current_block ? current_block->self : ID(0);
						break;
					}
				}
				else if (instr == NonSemanticShaderDebugInfo100DebugLocalVariable)
				{
					if (length < 11)
						SPIRV_CROSS_THROW("Invalid arguments for ShaderDebugInfo100DebugLocalVariable");
					auto &lvar = set<SPIRDebugLocalVariable>(ops[1]);
					lvar.name_id = ops[4];
				}
				else if (instr == NonSemanticShaderDebugInfo100DebugDeclare)
				{
					if (length < 7)
						SPIRV_CROSS_THROW("Invalid arguments for ShaderDebugInfo100DebugDeclare");
					auto &lvar = get<SPIRDebugLocalVariable>(ops[4]);
					auto &var = get<SPIRVariable>(ops[5]);
					var.debug_local_variables.push_back(lvar.self);
				}
			}
		}
		break;
	}

	case OpEntryPoint:
	{
		if (!is_valid_execution_model(ops[0]))
			SPIRV_CROSS_THROW("Invalid execution model in SPIR-V module.");
		auto itr =
		    ir.entry_points.insert(make_pair(ops[1], SPIREntryPoint(ops[1], static_cast<ExecutionModel>(ops[0]),
		                                                            extract_string(ir.spirv, instruction.offset + 2))));
		auto &e = itr.first->second;

		// Strings need nul-terminator and consume the whole word.
		uint32_t strlen_words = uint32_t((e.name.size() + 1 + 3) >> 2);

		for (uint32_t i = strlen_words + 2; i < instruction.length; i++)
		{
			if (ops[i] >= ir.ids.size())
				SPIRV_CROSS_THROW("OpEntryPoint interface ID " + std::to_string(ops[i]) +
				                  " is out of range for the declared ID bound " +
				                  std::to_string(ir.ids.size()) + ".");
			e.interface_variables.push_back(ops[i]);
		}

		// Set the name of the entry point in case OpName is not provided later.
		ir.set_name(ops[1], e.name);

		// If we don't have an entry, make the first one our "default".
		if (!ir.default_entry_point)
			ir.default_entry_point = ops[1];
		break;
	}

	case OpExecutionMode:
	{
		auto &execution = ir.entry_points[ops[0]];
		if (!is_valid_execution_mode(ops[1]))
			SPIRV_CROSS_THROW("Invalid execution mode in SPIR-V module.");
		auto mode = static_cast<ExecutionMode>(ops[1]);
		execution.flags.set(mode);

		// Modes whose literal operands are consumed below.
		if ((mode == ExecutionModeLocalSize && length < 5) ||
		    ((mode == ExecutionModeInvocations || mode == ExecutionModeOutputVertices ||
		      mode == ExecutionModeOutputPrimitivesEXT || mode == ExecutionModeSignedZeroInfNanPreserve) &&
		     length < 3))
			SPIRV_CROSS_THROW("Too few operands for execution mode in SPIR-V module.");

		switch (mode)
		{
		case ExecutionModeInvocations:
			execution.invocations = ops[2];
			break;

		case ExecutionModeLocalSize:
			execution.workgroup_size.x = ops[2];
			execution.workgroup_size.y = ops[3];
			execution.workgroup_size.z = ops[4];
			break;

		case ExecutionModeOutputVertices:
			execution.output_vertices = ops[2];
			break;

		case ExecutionModeOutputPrimitivesEXT:
			execution.output_primitives = ops[2];
			break;

		case ExecutionModeSignedZeroInfNanPreserve:
			switch (ops[2])
			{
			case 8:
				execution.signed_zero_inf_nan_preserve_8 = true;
				break;

			case 16:
				execution.signed_zero_inf_nan_preserve_16 = true;
				break;

			case 32:
				execution.signed_zero_inf_nan_preserve_32 = true;
				break;

			case 64:
				execution.signed_zero_inf_nan_preserve_64 = true;
				break;

			default:
				SPIRV_CROSS_THROW("Invalid bit-width for SignedZeroInfNanPreserve.");
			}
			break;

		default:
			break;
		}
		break;
	}

	case OpExecutionModeId:
	{
		auto &execution = ir.entry_points[ops[0]];
		if (!is_valid_execution_mode(ops[1]))
			SPIRV_CROSS_THROW("Invalid execution mode in SPIR-V module.");
		auto mode = static_cast<ExecutionMode>(ops[1]);
		execution.flags.set(mode);

		// Modes whose operands are consumed below.
		if ((mode == ExecutionModeLocalSizeId && length < 5) ||
		    (mode == ExecutionModeFPFastMathDefault && length < 4))
			SPIRV_CROSS_THROW("Too few operands for execution mode in SPIR-V module.");

		switch (mode)
		{
		case ExecutionModeLocalSizeId:
			execution.workgroup_size.id_x = ops[2];
			execution.workgroup_size.id_y = ops[3];
			execution.workgroup_size.id_z = ops[4];
			break;

		case ExecutionModeFPFastMathDefault:
			execution.fp_fast_math_defaults[ops[2]] = ops[3];
			break;

		default:
			break;
		}
		break;
	}

	case OpName:
	{
		uint32_t id = ops[0];
		ir.set_name(id, extract_string(ir.spirv, instruction.offset + 1));
		break;
	}

	case OpMemberName:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		ir.set_member_name(id, member, extract_string(ir.spirv, instruction.offset + 2));
		break;
	}

	case OpDecorationGroup:
	{
		// Noop, this simply means an ID should be a collector of decorations.
		// The meta array is already a flat array of decorations which will contain the relevant decorations.
		break;
	}

	case OpGroupDecorate:
	{
		uint32_t group_id = ops[0];
		auto &decorations = ir.meta[group_id].decoration;
		auto &flags = decorations.decoration_flags;

		// Copies decorations from one ID to another. Only copy decorations which are set in the group,
		// i.e., we cannot just copy the meta structure directly.
		for (uint32_t i = 1; i < length; i++)
		{
			uint32_t target = ops[i];
			flags.for_each_bit([&](uint32_t bit) {
				auto decoration = static_cast<Decoration>(bit);

				if (decoration_is_string(decoration))
				{
					ir.set_decoration_string(target, decoration, ir.get_decoration_string(group_id, decoration));
				}
				else
				{
					ir.meta[target].decoration_word_offset[decoration] =
					    ir.meta[group_id].decoration_word_offset[decoration];
					ir.set_decoration(target, decoration, ir.get_decoration(group_id, decoration));
				}
			});
		}
		break;
	}

	case OpGroupMemberDecorate:
	{
		uint32_t group_id = ops[0];
		auto &flags = ir.meta[group_id].decoration.decoration_flags;

		// Copies decorations from one ID to another. Only copy decorations which are set in the group,
		// i.e., we cannot just copy the meta structure directly.
		for (uint32_t i = 1; i + 1 < length; i += 2)
		{
			uint32_t target = ops[i + 0];
			uint32_t index = ops[i + 1];
			flags.for_each_bit([&](uint32_t bit) {
				auto decoration = static_cast<Decoration>(bit);

				if (decoration_is_string(decoration))
					ir.set_member_decoration_string(target, index, decoration,
					                                ir.get_decoration_string(group_id, decoration));
				else
					ir.set_member_decoration(target, index, decoration, ir.get_decoration(group_id, decoration));
			});
		}
		break;
	}

	case OpDecorate:
	case OpDecorateId:
	{
		// OpDecorateId technically supports an array of arguments, but our only supported decorations are single uint,
		// so merge decorate and decorate-id here.
		uint32_t id = ops[0];

		if (!is_valid_decoration(ops[1]))
			SPIRV_CROSS_THROW("Invalid decoration in SPIR-V module.");
		auto decoration = static_cast<Decoration>(ops[1]);
		if (length >= 3)
		{
			ir.meta[id].decoration_word_offset[decoration] = uint32_t(&ops[2] - ir.spirv.data());
			ir.set_decoration(id, decoration, ops[2]);
		}
		else
			ir.set_decoration(id, decoration);

		// Track exported functions so we can compile library modules that have no OpEntryPoint.
		// LinkageAttributes layout: literal-string (variable words) followed by LinkageType.
		if (decoration == DecorationLinkageAttributes && length >= 4 &&
		    static_cast<LinkageType>(ops[length - 1]) == LinkageTypeExport)
		{
			ir.library_exports.push_back(id);

			// If OpName was stripped (e.g. by spirv-opt --strip-debug), fall back
			// to the linkage name so the emitted function keeps its export name.
			if (ir.get_name(id).empty())
				ir.set_name(id, extract_string(ir.spirv, instruction.offset + 2));
		}

		break;
	}

	case OpDecorateStringGOOGLE:
	{
		uint32_t id = ops[0];
		if (!is_valid_decoration(ops[1]))
			SPIRV_CROSS_THROW("Invalid decoration in SPIR-V module.");
		auto decoration = static_cast<Decoration>(ops[1]);
		ir.set_decoration_string(id, decoration, extract_string(ir.spirv, instruction.offset + 2));
		break;
	}

	case OpMemberDecorate:
	case OpMemberDecorateIdEXT:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		if (!is_valid_decoration(ops[2]))
			SPIRV_CROSS_THROW("Invalid decoration in SPIR-V module.");
		auto decoration = static_cast<Decoration>(ops[2]);
		if (length >= 4)
			ir.set_member_decoration(id, member, decoration, ops[3]);
		else
			ir.set_member_decoration(id, member, decoration);
		break;
	}

	case OpMemberDecorateStringGOOGLE:
	{
		uint32_t id = ops[0];
		uint32_t member = ops[1];
		if (!is_valid_decoration(ops[2]))
			SPIRV_CROSS_THROW("Invalid decoration in SPIR-V module.");
		auto decoration = static_cast<Decoration>(ops[2]);
		ir.set_member_decoration_string(id, member, decoration, extract_string(ir.spirv, instruction.offset + 3));
		break;
	}

	// Build up basic types.
	case OpTypeVoid:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Void;
		break;
	}

	case OpTypeBool:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Boolean;
		type.width = 1;
		break;
	}

	case OpTypeFloat:
	{
		uint32_t id = ops[0];
		uint32_t width = ops[1];
		auto &type = set<SPIRType>(id, op);

		if (width != 16 && width != 8 && length > 2)
			SPIRV_CROSS_THROW("Unrecognized FP encoding mode for OpTypeFloat.");

		if (width == 64)
			type.basetype = SPIRType::Double;
		else if (width == 32)
			type.basetype = SPIRType::Float;
		else if (width == 16)
		{
			if (length > 2)
			{
				if (ops[2] == FPEncodingBFloat16KHR)
					type.basetype = SPIRType::BFloat16;
				else
					SPIRV_CROSS_THROW("Unrecognized encoding for OpTypeFloat 16.");
			}
			else
				type.basetype = SPIRType::Half;
		}
		else if (width == 8)
		{
			if (length < 3)
				SPIRV_CROSS_THROW("Missing encoding for OpTypeFloat 8.");
			else if (ops[2] == FPEncodingFloat8E4M3EXT)
				type.basetype = SPIRType::FloatE4M3;
			else if (ops[2] == FPEncodingFloat8E5M2EXT)
				type.basetype = SPIRType::FloatE5M2;
			else
				SPIRV_CROSS_THROW("Invalid encoding for OpTypeFloat 8.");
		}
		else
			SPIRV_CROSS_THROW("Unrecognized bit-width of floating point type.");
		type.width = width;
		break;
	}

	case OpTypeInt:
	{
		uint32_t id = ops[0];
		uint32_t width = ops[1];
		bool signedness = ops[2] != 0;
		auto &type = set<SPIRType>(id, op);
		type.basetype = signedness ? to_signed_basetype(width) : to_unsigned_basetype(width);
		type.width = width;
		break;
	}

	// Build composite types by "inheriting".
	// NOTE: The self member is also copied! For pointers and array modifiers this is a good thing
	// since we can refer to decorations on pointee classes which is needed for UBO/SSBO, I/O blocks in geometry/tess etc.
	case OpTypeVector:
	{
		uint32_t id = ops[0];
		uint32_t vecsize = ops[2];

		if (ops[1] == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &base = get<SPIRType>(ops[1]);
		auto &vecbase = set<SPIRType>(id, base);

		vecbase.op = op;
		vecbase.vecsize = vecsize;
		vecbase.self = id;
		vecbase.parent_type = ops[1];
		break;
	}

	case OpTypeMatrix:
	{
		uint32_t id = ops[0];
		uint32_t colcount = ops[2];

		if (ops[1] == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &base = get<SPIRType>(ops[1]);
		auto &matrixbase = set<SPIRType>(id, base);

		matrixbase.op = op;
		matrixbase.columns = colcount;
		matrixbase.self = id;
		matrixbase.parent_type = ops[1];
		break;
	}

	case OpTypeCooperativeMatrixKHR:
	{
		uint32_t id = ops[0];
		if (ops[1] == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &base = get<SPIRType>(ops[1]);
		auto &matrixbase = set<SPIRType>(id, base);

		matrixbase.op = op;
		matrixbase.ext.cooperative.scope_id = ops[2];
		matrixbase.ext.cooperative.rows_id = ops[3];
		matrixbase.ext.cooperative.columns_id = ops[4];
		matrixbase.ext.cooperative.use_id = ops[5];
		matrixbase.self = id;
		matrixbase.parent_type = ops[1];
		break;
	}

	case OpTypeCooperativeVectorNV:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);

		type.basetype = SPIRType::CoopVecNV;
		type.op = op;
		type.ext.coopVecNV.component_type_id = ops[1];
		type.ext.coopVecNV.component_count_id = ops[2];
		type.parent_type = ops[1];

		// CoopVec-Nv can be used with integer operations like SMax where
		// where spirv-opt does explicit checks on integer bitwidth
		auto component_type = get<SPIRType>(type.ext.coopVecNV.component_type_id);
		type.width = component_type.width;
		break;
	}

	case OpTypeArray:
	{
		uint32_t id = ops[0];
		uint32_t tid = ops[1];
		if (tid == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &base = get<SPIRType>(tid);
		auto &arraybase = set<SPIRType>(id, base);

		arraybase.op = op;
		arraybase.parent_type = tid;

		uint32_t cid = ops[2];
		ir.mark_used_as_array_length(cid);
		auto *c = maybe_get<SPIRConstant>(cid);
		bool literal = c && !c->specialization;

		// We're copying type information into Array types, so we'll need a fixup for any physical pointer
		// references.
		if (base.forward_pointer)
			forward_pointer_fixups.push_back({ id, tid });

		arraybase.array_size_literal.push_back(literal);
		arraybase.array.push_back(literal ? c->scalar() : cid);

		// .self resolves down to non-array/non-pointer type.
		arraybase.self = base.self;
		break;
	}

	case OpTypeRuntimeArray:
	{
		uint32_t id = ops[0];

		if (ops[1] == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &base = get<SPIRType>(ops[1]);
		auto &arraybase = set<SPIRType>(id, base);

		// We're copying type information into Array types, so we'll need a fixup for any physical pointer
		// references.
		if (base.forward_pointer)
			forward_pointer_fixups.push_back({ id, ops[1] });

		arraybase.op = op;
		arraybase.array.push_back(0);
		arraybase.array_size_literal.push_back(true);
		arraybase.parent_type = ops[1];

		// .self resolves down to non-array/non-pointer type.
		arraybase.self = base.self;
		break;
	}

	case OpTypeImage:
	{
		uint32_t id = ops[0];
		if (!is_valid_dim(ops[2]) || !is_valid_image_format(ops[7]) ||
		    (length >= 9 && !is_valid_access_qualifier(ops[8])))
			SPIRV_CROSS_THROW("Invalid image type in SPIR-V module.");
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Image;
		type.image.type = ops[1];
		type.image.dim = static_cast<Dim>(ops[2]);
		type.image.depth = ops[3] == 1;
		type.image.arrayed = ops[4] != 0;
		type.image.ms = ops[5] != 0;
		type.image.sampled = ops[6];
		type.image.format = static_cast<ImageFormat>(ops[7]);
		type.image.access = (length >= 9) ? static_cast<AccessQualifier>(ops[8]) : AccessQualifierMax;
		break;
	}

	case OpTypeSampledImage:
	{
		uint32_t id = ops[0];
		uint32_t imagetype = ops[1];
		if (imagetype == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto &type = set<SPIRType>(id, op);
		type = get<SPIRType>(imagetype);
		type.basetype = SPIRType::SampledImage;
		type.self = id;
		break;
	}

	case OpTypeSampler:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Sampler;
		break;
	}

	case OpTypeUntypedPointerKHR:
	case OpTypePointer:
	{
		uint32_t id = ops[0];

		if (!is_valid_storage_class(ops[1]))
			SPIRV_CROSS_THROW("Invalid storage class in SPIR-V module.");

		// Very rarely, we might receive a FunctionPrototype here.
		// We won't be able to compile it, but we shouldn't crash when parsing.
		// We should be able to reflect.
		if (op == OpTypePointer && ops[2] == id)
			SPIRV_CROSS_THROW("Self-referential type declaration is not allowed.");
		auto *base = op == OpTypePointer ? maybe_get<SPIRType>(ops[2]) : nullptr;
		auto &ptrbase = set<SPIRType>(id, op);

		if (base)
		{
			ptrbase = *base;
			ptrbase.op = op;
		}

		ptrbase.pointer = true;
		ptrbase.pointer_depth++;
		ptrbase.storage = static_cast<StorageClass>(ops[1]);

		if (ptrbase.storage == StorageClassAtomicCounter)
			ptrbase.basetype = SPIRType::AtomicCounter;

		if (base && base->forward_pointer)
			forward_pointer_fixups.push_back({ id, ops[2] });

		if (op == OpTypePointer)
			ptrbase.parent_type = ops[2];
		else
			ptrbase.basetype = SPIRType::Void;

		// Do NOT set ptrbase.self!
		break;
	}

	case OpTypeForwardPointer:
	{
		uint32_t id = ops[0];
		if (!is_valid_storage_class(ops[1]))
			SPIRV_CROSS_THROW("Invalid storage class in SPIR-V module.");
		auto &ptrbase = set<SPIRType>(id, op);
		ptrbase.pointer = true;
		ptrbase.pointer_depth++;
		ptrbase.storage = static_cast<StorageClass>(ops[1]);
		ptrbase.forward_pointer = true;

		if (ptrbase.storage == StorageClassAtomicCounter)
			ptrbase.basetype = SPIRType::AtomicCounter;

		break;
	}

	case OpTypeStruct:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Struct;
		for (uint32_t i = 1; i < length; i++)
			type.member_types.push_back(ops[i]);

		// Check if we have seen this struct type before, with just different
		// decorations.
		//
		// Add workaround for issue #17 as well by looking at OpName for the struct
		// types, which we shouldn't normally do.
		// We should not normally have to consider type aliases like this to begin with
		// however ... glslang issues #304, #307 cover this.

		// For stripped names, never consider struct type aliasing.
		// We risk declaring the same struct multiple times, but type-punning is not allowed
		// so this is safe.
		bool consider_aliasing = !ir.get_name(type.self).empty();
		if (consider_aliasing)
		{
			for (auto &other : global_struct_cache)
			{
				if (ir.get_name(type.self) == ir.get_name(other) &&
				    types_are_logically_equivalent(type, get<SPIRType>(other)))
				{
					type.type_alias = other;
					break;
				}
			}

			if (type.type_alias == TypeID(0))
				global_struct_cache.push_back(id);
		}
		break;
	}

	case OpTypeFunction:
	{
		uint32_t id = ops[0];
		uint32_t ret = ops[1];

		auto &func = set<SPIRFunctionPrototype>(id, ret);
		for (uint32_t i = 2; i < length; i++)
			func.parameter_types.push_back(ops[i]);
		break;
	}

	case OpTypeAccelerationStructureKHR:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::AccelerationStructure;
		break;
	}

	case OpTypeRayQueryKHR:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::RayQuery;
		break;
	}

	case OpTypeTensorARM:
	{
		uint32_t id = ops[0];
		auto &type = set<SPIRType>(id, op);
		type.basetype = SPIRType::Tensor;
		type.ext.tensor = {};
		type.ext.tensor.type = ops[1];
		if (length >= 3)
			type.ext.tensor.rank = ops[2];
		if (length >= 4)
			type.ext.tensor.shape = ops[3];
		break;
	}

	// Variable declaration
	// All variables are essentially pointers with a storage qualifier.
	case OpVariable:
	{
		uint32_t type = ops[0];
		uint32_t id = ops[1];
		if (!is_valid_storage_class(ops[2]))
			SPIRV_CROSS_THROW("Invalid storage class in SPIR-V module.");
		auto storage = static_cast<StorageClass>(ops[2]);
		uint32_t initializer = length == 4 ? ops[3] : 0;

		if (storage == StorageClassFunction)
		{
			if (!current_function)
				SPIRV_CROSS_THROW("No function currently in scope");
			current_function->add_local_variable(id);
		}

		set<SPIRVariable>(id, type, storage, initializer);
		break;
	}

	case OpUntypedVariableKHR:
	{
		uint32_t type = ops[0];
		uint32_t id = ops[1];
		if (!is_valid_storage_class(ops[2]))
			SPIRV_CROSS_THROW("Invalid storage class in SPIR-V module.");
		auto storage = static_cast<StorageClass>(ops[2]);
		uint32_t data_type = length >= 4 ? ops[3] : 0;
		uint32_t initializer = length >= 5 ? ops[4] : 0;

		if (storage == StorageClassFunction)
		{
			if (!current_function)
				SPIRV_CROSS_THROW("No function currently in scope");
			current_function->add_local_variable(id);
		}

		auto &v = set<SPIRVariable>(id, type, storage, initializer);
		v.untyped = true;
		v.untyped_alloca_type = data_type;
		break;
	}

	// OpPhi
	// OpPhi is a fairly magical opcode.
	// It selects temporary variables based on which parent block we *came from*.
	// In high-level languages we can "de-SSA" by creating a function local, and flush out temporaries to this function-local
	// variable to emulate SSA Phi.
	case OpPhi:
	{
		if (!current_function)
			SPIRV_CROSS_THROW("No function currently in scope");
		if (!current_block)
			SPIRV_CROSS_THROW("No block currently in scope");

		uint32_t result_type = ops[0];
		uint32_t id = ops[1];

		// Instead of a temporary, create a new function-wide temporary with this ID instead.
		auto &var = set<SPIRVariable>(id, result_type, StorageClassFunction);
		var.phi_variable = true;

		current_function->add_local_variable(id);

		for (uint32_t i = 2; i + 2 <= length; i += 2)
			current_block->phi_variables.push_back({ ops[i], ops[i + 1], id });
		break;
	}

	// Constants
	case OpSpecConstant:
	case OpConstant:
	case OpConstantCompositeReplicateEXT:
	case OpSpecConstantCompositeReplicateEXT:
	{
		uint32_t id = ops[1];
		auto &type = get<SPIRType>(ops[0]);
		if (op == OpConstantCompositeReplicateEXT || op == OpSpecConstantCompositeReplicateEXT)
		{
			auto subconstant = uint32_t(ops[2]);
			set<SPIRConstant>(id, ops[0], &subconstant, 1, op == OpSpecConstantCompositeReplicateEXT, true);
		}
		else
		{
			if (type.width > 32)
			{
				if (length < 4)
					SPIRV_CROSS_THROW("OpConstant with a 64-bit type requires two literal words.");
				set<SPIRConstant>(id, ops[0], ops[2] | (uint64_t(ops[3]) << 32), op == OpSpecConstant);
			}
			else
				set<SPIRConstant>(id, ops[0], ops[2], op == OpSpecConstant);
		}
		break;
	}

	case OpSpecConstantFalse:
	case OpConstantFalse:
	{
		uint32_t id = ops[1];
		set<SPIRConstant>(id, ops[0], uint32_t(0), op == OpSpecConstantFalse);
		break;
	}

	case OpSpecConstantTrue:
	case OpConstantTrue:
	{
		uint32_t id = ops[1];
		set<SPIRConstant>(id, ops[0], uint32_t(1), op == OpSpecConstantTrue);
		break;
	}

	case OpConstantNull:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];
		ir.make_constant_null(id, type, true);
		break;
	}

	case OpSpecConstantComposite:
	case OpConstantComposite:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];

		auto &ctype = get<SPIRType>(type);
		uint32_t elements = length - 2;

		// We can have constants which are structs and arrays.
		// In this case, our SPIRConstant will be a list of other SPIRConstant ids which we
		// can refer to.
		if (ctype.basetype == SPIRType::Struct || !ctype.array.empty() || elements > 4)
		{
			set<SPIRConstant>(id, type, ops + 2, length - 2, op == OpSpecConstantComposite);
		}
		else
		{
			SPIRConstant remapped_constant_ops[4];
			const SPIRConstant *c[4];
			for (uint32_t i = 0; i < elements; i++)
			{
				if (ops[2 + i] == id)
					SPIRV_CROSS_THROW("Self-referential constant declaration is not allowed.");
				// Specialization constants operations can also be part of this.
				// We do not know their value, so any attempt to query SPIRConstant later
				// will fail. We can only propagate the ID of the expression and use to_expression on it.
				auto *constant_op = maybe_get<SPIRConstantOp>(ops[2 + i]);
				auto *undef_op = maybe_get<SPIRUndef>(ops[2 + i]);
				if (constant_op)
				{
					if (op == OpConstantComposite)
						SPIRV_CROSS_THROW("Specialization constant operation used in OpConstantComposite.");

					remapped_constant_ops[i].make_null(get<SPIRType>(constant_op->basetype));
					remapped_constant_ops[i].self = constant_op->self;
					remapped_constant_ops[i].constant_type = constant_op->basetype;
					remapped_constant_ops[i].specialization = true;
					c[i] = &remapped_constant_ops[i];
				}
				else if (undef_op)
				{
					// Undefined, just pick 0.
					remapped_constant_ops[i].make_null(get<SPIRType>(undef_op->basetype));
					remapped_constant_ops[i].constant_type = undef_op->basetype;
					c[i] = &remapped_constant_ops[i];
				}
				else
					c[i] = &get<SPIRConstant>(ops[2 + i]);
			}
			set<SPIRConstant>(id, type, c, elements, op == OpSpecConstantComposite);
		}
		break;
	}

	case OpConstantSizeOfEXT:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];
		auto &c = set<SPIRConstant>(id, type);
		c.size_of_type = ops[2];
		break;
	}

	case OpConstantDataKHR:
	case OpSpecConstantDataKHR:
	{
		uint32_t id = ops[1];
		uint32_t type = ops[0];
		set<SPIRConstantData>(id, type, ops + 2, length - 2, op == OpSpecConstantDataKHR);
		break;
	}

	case OpTypeBufferEXT:
	{
		uint32_t type = ops[0];
		if (!is_valid_storage_class(ops[1]))
			SPIRV_CROSS_THROW("Invalid storage class in SPIR-V module.");
		auto &t = set<SPIRType>(type, OpTypeBufferEXT);
		t.basetype = SPIRType::DescriptorHeapBuffer;
		t.ext.descriptor_heap_buffer.storage = static_cast<StorageClass>(ops[1]);
		break;
	}

	// Functions
	case OpFunction:
	{
		uint32_t res = ops[0];
		uint32_t id = ops[1];
		uint32_t control = ops[2];
		uint32_t type = ops[3];

		if (current_function)
			SPIRV_CROSS_THROW("Must end a function before starting a new one!");

		current_function = &set<SPIRFunction>(id, res, control, type);
		break;
	}

	case OpFunctionParameter:
	{
		uint32_t type = ops[0];
		uint32_t id = ops[1];

		if (!current_function)
			SPIRV_CROSS_THROW("Must be in a function!");

		current_function->add_parameter(type, id);
		set<SPIRVariable>(id, type, StorageClassFunction);
		break;
	}

	case OpFunctionEnd:
	{
		if (current_block)
		{
			// Very specific error message, but seems to come up quite often.
			SPIRV_CROSS_THROW(
			    "Cannot end a function before ending the current block.\n"
			    "Likely cause: If this SPIR-V was created from glslang HLSL, make sure the entry point is valid.");
		}
		current_function = nullptr;
		break;
	}

	// Blocks
	case OpLabel:
	{
		// OpLabel always starts a block.
		if (!current_function)
			SPIRV_CROSS_THROW("Blocks cannot exist outside functions!");

		uint32_t id = ops[0];

		current_function->blocks.push_back(id);
		if (!current_function->entry_block)
			current_function->entry_block = id;

		if (current_block)
			SPIRV_CROSS_THROW("Cannot start a block before ending the current block.");

		current_block = &set<SPIRBlock>(id);
		break;
	}

	// Branch instructions end blocks.
	case OpBranch:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		uint32_t target = ops[0];
		current_block->terminator = SPIRBlock::Direct;
		current_block->next_block = target;
		current_block = nullptr;
		break;
	}

	case OpBranchConditional:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		current_block->condition = ops[0];
		current_block->true_block = ops[1];
		current_block->false_block = ops[2];

		current_block->terminator = SPIRBlock::Select;

		if (current_block->true_block == current_block->false_block)
		{
			// Bogus conditional, translate to a direct branch.
			// Avoids some ugly edge cases later when analyzing CFGs.

			// There are some super jank cases where the merge block is different from the true/false,
			// and later branches can "break" out of the selection construct this way.
			// This is complete nonsense, but CTS hits this case.
			// In this scenario, we should see the selection construct as more of a Switch with one default case.
			// The problem here is that this breaks any attempt to break out of outer switch statements,
			// but it's theoretically solvable if this ever comes up using the ladder breaking system ...

			if (current_block->true_block != current_block->next_block &&
			    current_block->merge == SPIRBlock::MergeSelection)
			{
				uint32_t ids = ir.increase_bound_by(2);

				auto &type = set<SPIRType>(ids, OpTypeInt);
				type.basetype = SPIRType::Int;
				type.width = 32;
				auto &c = set<SPIRConstant>(ids + 1, ids);

				current_block->condition = c.self;
				current_block->default_block = current_block->true_block;
				current_block->terminator = SPIRBlock::MultiSelect;
				block_meta_flags(current_block->next_block) &= ~ParsedIR::BLOCK_META_SELECTION_MERGE_BIT;
				block_meta_flags(current_block->next_block) |= ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT;
			}
			else
			{
				// Collapse loops if we have to.
				bool collapsed_loop = current_block->true_block == current_block->merge_block &&
				                      current_block->merge == SPIRBlock::MergeLoop;

				if (collapsed_loop)
				{
					block_meta_flags(current_block->merge_block) &= ~ParsedIR::BLOCK_META_LOOP_MERGE_BIT;
					block_meta_flags(current_block->continue_block) &= ~ParsedIR::BLOCK_META_CONTINUE_BIT;
				}

				current_block->next_block = current_block->true_block;
				current_block->condition = 0;
				current_block->true_block = 0;
				current_block->false_block = 0;
				current_block->merge_block = 0;
				current_block->merge = SPIRBlock::MergeNone;
				current_block->terminator = SPIRBlock::Direct;
			}
		}

		current_block = nullptr;
		break;
	}

	case OpSwitch:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");

		current_block->terminator = SPIRBlock::MultiSelect;

		current_block->condition = ops[0];
		current_block->default_block = ops[1];

		uint32_t remaining_ops = length - 2;
		if ((remaining_ops % 2) == 0)
		{
			for (uint32_t i = 2; i + 2 <= length; i += 2)
				current_block->cases_32bit.push_back({ ops[i], ops[i + 1] });
		}

		if ((remaining_ops % 3) == 0)
		{
			for (uint32_t i = 2; i + 3 <= length; i += 3)
			{
				uint64_t value = (static_cast<uint64_t>(ops[i + 1]) << 32) | ops[i];
				current_block->cases_64bit.push_back({ value, ops[i + 2] });
			}
		}

		// If we jump to next block, make it break instead since we're inside a switch case block at that point.
		block_meta_flags(current_block->next_block) |= ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT;

		current_block = nullptr;
		break;
	}

	case OpKill:
	case OpTerminateInvocation:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Kill;
		current_block = nullptr;
		break;
	}

	case OpTerminateRayKHR:
		// NV variant is not a terminator.
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::TerminateRay;
		current_block = nullptr;
		break;

	case OpIgnoreIntersectionKHR:
		// NV variant is not a terminator.
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::IgnoreIntersection;
		current_block = nullptr;
		break;

	case OpEmitMeshTasksEXT:
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::EmitMeshTasks;
		for (uint32_t i = 0; i < 3; i++)
			current_block->mesh.groups[i] = ops[i];
		current_block->mesh.payload = length >= 4 ? ops[3] : 0;
		current_block = nullptr;
		// Currently glslang is bugged and does not treat EmitMeshTasksEXT as a terminator.
		ignore_trailing_block_opcodes = true;
		break;

	case OpAbortKHR:
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::ShaderAbort;
		current_block->shader_abort.physical_type = ops[0];
		current_block->shader_abort.payload = ops[1];
		current_block = nullptr;
		break;

	case OpReturn:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Return;
		current_block = nullptr;
		break;
	}

	case OpReturnValue:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Return;
		current_block->return_value = ops[0];
		current_block = nullptr;
		break;
	}

	case OpUnreachable:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to end a non-existing block.");
		current_block->terminator = SPIRBlock::Unreachable;
		current_block = nullptr;
		break;
	}

	case OpSelectionMerge:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to modify a non-existing block.");

		current_block->next_block = ops[0];
		current_block->merge = SPIRBlock::MergeSelection;
		block_meta_flags(current_block->next_block) |= ParsedIR::BLOCK_META_SELECTION_MERGE_BIT;

		if (length >= 2)
		{
			if (ops[1] & SelectionControlFlattenMask)
				current_block->hint = SPIRBlock::HintFlatten;
			else if (ops[1] & SelectionControlDontFlattenMask)
				current_block->hint = SPIRBlock::HintDontFlatten;
		}
		break;
	}

	case OpLoopMerge:
	{
		if (!current_block)
			SPIRV_CROSS_THROW("Trying to modify a non-existing block.");

		current_block->merge_block = ops[0];
		current_block->continue_block = ops[1];
		current_block->merge = SPIRBlock::MergeLoop;

		block_meta_flags(current_block->self) |= ParsedIR::BLOCK_META_LOOP_HEADER_BIT;
		block_meta_flags(current_block->merge_block) |= ParsedIR::BLOCK_META_LOOP_MERGE_BIT;

		ir.continue_block_to_loop_header[current_block->continue_block] = BlockID(current_block->self);

		// Don't add loop headers to continue blocks,
		// which would make it impossible branch into the loop header since
		// they are treated as continues.
		if (current_block->continue_block != BlockID(current_block->self))
			block_meta_flags(current_block->continue_block) |= ParsedIR::BLOCK_META_CONTINUE_BIT;

		if (length >= 3)
		{
			if (ops[2] & LoopControlUnrollMask)
				current_block->hint = SPIRBlock::HintUnroll;
			else if (ops[2] & LoopControlDontUnrollMask)
				current_block->hint = SPIRBlock::HintDontUnroll;
		}
		break;
	}

	case OpSpecConstantOp:
	{
		if (length < 3)
			SPIRV_CROSS_THROW("OpSpecConstantOp not enough arguments.");

		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		if (!is_known_opcode(ops[2]))
			SPIRV_CROSS_THROW("Unknown opcode in OpSpecConstantOp.");
		auto spec_op = static_cast<Op>(ops[2]);

		set<SPIRConstantOp>(id, result_type, spec_op, ops + 3, length - 3);
		break;
	}

	case OpLine:
	{
		// OpLine might come at global scope, but we don't care about those since they will not be declared in any
		// meaningful correct order.
		// Ignore all OpLine directives which live outside a function.
		if (current_block)
			current_block->ops.push_back(instruction);

		// Line directives may arrive before first OpLabel.
		// Treat this as the line of the function declaration,
		// so warnings for arguments can propagate properly.
		if (current_function)
		{
			// Store the first one we find and emit it before creating the function prototype.
			if (current_function->entry_line.file_id == 0)
			{
				current_function->entry_line.file_id = ops[0];
				current_function->entry_line.line_literal = ops[1];
			}
		}

		uint32_t file = ops[0];
		uint32_t line = ops[1];

		for (auto &source : ir.sources)
		{
			if (source.file_id == file)
			{
				source.line_markers.emplace_back();
				auto &marker = source.line_markers.back();
				marker.line = line;
				marker.offset = instruction.offset - 1;
				marker.function_id = current_function ? current_function->self : ID(0);
				marker.block_id = current_block ? current_block->self : ID(0);
				break;
			}
		}

		break;
	}

	case OpNoLine:
	{
		// OpNoLine might come at global scope.
		if (current_block)
			current_block->ops.push_back(instruction);
		break;
	}

	// Actual opcodes.
	default:
	{
		if (length >= 2)
		{
			const auto *type = maybe_get<SPIRType>(ops[0]);
			if (type)
				ir.load_type_width.insert({ ops[1], type->width });
		}

		if (!current_block)
			SPIRV_CROSS_THROW("Currently no block to insert opcode.");

		current_block->ops.push_back(instruction);
		break;
	}
	}
}

bool Parser::types_are_logically_equivalent(const SPIRType &a, const SPIRType &b) const
{
	unordered_set<uint64_t> visited;
	return types_are_logically_equivalent(a, b, visited);
}

bool Parser::types_are_logically_equivalent(const SPIRType &a, const SPIRType &b,
                                            unordered_set<uint64_t> &visited) const
{
	if (a.basetype != b.basetype)
		return false;
	if (a.width != b.width)
		return false;
	if (a.vecsize != b.vecsize)
		return false;
	if (a.columns != b.columns)
		return false;
	if (a.array.size() != b.array.size())
		return false;

	size_t array_count = a.array.size();
	if (array_count && memcmp(a.array.data(), b.array.data(), array_count * sizeof(uint32_t)) != 0)
		return false;

	if (a.basetype == SPIRType::Image || a.basetype == SPIRType::SampledImage)
	{
		if (memcmp(&a.image, &b.image, sizeof(SPIRType::Image)) != 0)
			return false;
	}

	if (a.member_types.size() != b.member_types.size())
		return false;

	// Buffer references let a struct point back to itself. If we're already
	// comparing this pair, assume it matches and let the other members decide.
	if (!visited.insert((uint64_t(a.self) << 32) | uint64_t(b.self)).second)
		return true;

	size_t member_types = a.member_types.size();
	for (size_t i = 0; i < member_types; i++)
	{
		if (!types_are_logically_equivalent(get<SPIRType>(a.member_types[i]), get<SPIRType>(b.member_types[i]),
		                                    visited))
			return false;
	}

	return true;
}

bool Parser::variable_storage_is_aliased(const SPIRVariable &v) const
{
	auto &type = get<SPIRType>(v.basetype);

	auto *type_meta = ir.find_meta(type.self);

	bool ssbo = v.storage == StorageClassStorageBuffer ||
	            (type_meta && type_meta->decoration.decoration_flags.get(DecorationBufferBlock));
	bool image = type.basetype == SPIRType::Image;
	bool counter = type.basetype == SPIRType::AtomicCounter;

	bool is_restrict;
	if (ssbo)
		is_restrict = ir.get_buffer_block_flags(v).get(DecorationRestrict);
	else
		is_restrict = ir.has_decoration(v.self, DecorationRestrict);

	return !is_restrict && (ssbo || image || counter);
}
} // namespace SPIRV_CROSS_NAMESPACE
