// SPDX-License-Identifier: Apache-2.0
#include "vllm_quant_grouped_matmul_dequant_v310_tiling.h"
#include "log/ops_log.h"

namespace optiling {
namespace {
constexpr uint32_t DECODE_ROWS = 640;
constexpr uint32_t EXPERTS = 256;
constexpr uint32_t CORES = 8;
constexpr uint32_t HIDDEN = 2048;
constexpr uint32_t GATE_UP = 512;
constexpr uint32_t INTERMEDIATE = 256;
constexpr uint32_t NZ_M = 16;
constexpr uint32_t NZ_K = 32;
constexpr uint32_t GEMV_WEIGHT_K = 512;
constexpr uint32_t GEMV_WEIGHT_N = 64;
constexpr uint32_t UB_BYTES = 256 * 1024;
constexpr uint32_t ROW_SCALE_BYTES = NZ_M * sizeof(float);
// FP16 ND, FP16 ZN, and FP32 ND buffers for 16 rows, plus row scales.
constexpr uint32_t PREPARE_K_MAX =
    ((UB_BYTES - ROW_SCALE_BYTES) / (NZ_M * (2 * sizeof(uint16_t) + sizeof(float)))) / NZ_K * NZ_K;

bool ShapeIs(const gert::Shape& shape, std::initializer_list<int64_t> dimensions)
{
    if (shape.GetDimNum() != dimensions.size()) return false;
    size_t i = 0;
    for (const auto dimension : dimensions) {
        if (shape.GetDim(i++) != dimension) return false;
    }
    return true;
}

ge::graphStatus Tiling(gert::TilingContext* context)
{
    if (context->GetPlatformInfo() == nullptr) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // This is an AiCore (Cube + Vector) kernel on 310P, not a vector-only op.
    if (platform.GetCoreNumAic() != CORES) {
        OPS_LOG_E(context, "VllmQuantGroupedMatmulDequantV310 requires an eight-core 310P.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 5; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto& x = context->GetInputShape(0)->GetOriginShape();
    const auto& w = context->GetInputShape(1)->GetOriginShape();
    const bool gate = ShapeIs(x, {DECODE_ROWS, HIDDEN}) && ShapeIs(w, {EXPERTS, GATE_UP, HIDDEN});
    const bool down = ShapeIs(x, {DECODE_ROWS, INTERMEDIATE}) && ShapeIs(w, {EXPERTS, HIDDEN, INTERMEDIATE});
    if (!gate && !down) return ge::GRAPH_FAILED;
    const uint32_t k = gate ? HIDDEN : INTERMEDIATE;
    const uint32_t n = gate ? GATE_UP : HIDDEN;
    const ge::DataType types[] = {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT};
    for (size_t i = 0; i < 5; ++i) {
        const auto format = context->GetInputDesc(i)->GetStorageFormat();
        if (context->GetInputDesc(i)->GetDataType() != types[i] ||
            format != (i == 1 ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!ShapeIs(context->GetInputShape(2)->GetOriginShape(), {EXPERTS, n}) ||
        !ShapeIs(context->GetInputShape(3)->GetOriginShape(), {EXPERTS}) ||
        !ShapeIs(context->GetInputShape(4)->GetOriginShape(), {DECODE_ROWS})) {
        return ge::GRAPH_FAILED;
    }
    VllmQuantGroupedMatmulDequantV310TilingData data;
    data.set_CoreNum(0);
    data.set_originM(0);
    data.set_originN(0);
    data.set_originK(0);
    data.set_originE(0);
    data.set_originKAligned32(0);
    data.set_originKAligned512(0);
    data.set_fracM(0);
    data.set_fracN(0);
    data.set_fracK(0);
    data.set_tailM(0);
    data.set_singleCoreFracN(0);
    data.set_singleCoreFracNTail(0);
    data.set_processXKBaseNMax(0);
    data.set_perToken(0);
    data.set_dynamicQuant(0);
    data.set_smoothScale(0);
    data.set_isXScaleHalf(0);
    data.set_dynamicBaseK(0);
    data.set_dynamicBaseKTail(0);
    data.set_dynamicIterK(0);
    data.set_MCoreNum(0);
    data.set_NCoreNum(0);
    data.set_singleCoreM(0);
    data.set_singleCoreMTail(0);
    data.set_singleCoreN(0);
    data.set_singleCoreNTail(0);
    data.set_baseMNum(0);
    data.set_baseNNum(0);
    data.set_baseNNum_2(0);
    data.set_baseKNum(0);
    data.set_baseKNum_2(0);
    data.set_baseK(0);
    data.set_baseK_2(0);
    data.set_baseKTail(0);
    data.set_baseKTail_2(0);
    data.set_processXKloopPerfracM(0);
    data.set_processXKloop(0);
    data.set_processXKBaseN(0);
    data.set_processXKTailN(0);
    data.set_swiftGEMVThreshold(0);
    data.set_baseFracK(0);
    data.set_baseFracN(0);
    data.set_baseFracNL0C(0);
    data.set_ubBaseK(0);
    data.set_ubBaseKTail(0);
    data.set_ubIterK(0);
    data.set_ubKMask(0);
    data.set_CoreNum(CORES);
    data.set_originM(DECODE_ROWS);
    data.set_originN(n);
    data.set_originK(k);
    data.set_originE(EXPERTS);
    data.set_originKAligned32(k);
    data.set_originKAligned512((k + GATE_UP - 1) / GATE_UP * GATE_UP);
    data.set_fracM(DECODE_ROWS / NZ_M);
    data.set_fracN(n / NZ_M);
    data.set_fracK(k / NZ_K);
    data.set_tailM(NZ_M);
    data.set_singleCoreFracN(n / NZ_M / CORES);
    data.set_singleCoreFracNTail(CORES);
    data.set_processXKBaseNMax(PREPARE_K_MAX);
    // GEMV's default 512 x 64 INT8 tile fills one 32 KiB ping-pong
    // weight slot. The down kernel selects its validated 256 x 128 tile.
    data.set_baseFracK(GEMV_WEIGHT_K / NZ_K);
    data.set_baseFracN(GEMV_WEIGHT_N / NZ_M);
    data.set_perToken(1);
    const uint64_t offset = platform.GetLibApiWorkSpaceSize();
    data.set_workspaceOffset(offset);
    auto* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) return ge::GRAPH_FAILED;
    // Eight private 80-row slots, reused by the shared >80-row phase, plus
    // one 32-byte synchronization slot per core. No host read of group_list.
    workspaces[0] = offset + uint64_t(DECODE_ROWS) * k + CORES * NZ_K;
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->SetBlockDim(CORES);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(VllmQuantGroupedMatmulDequantV310).Tiling(Tiling);
}  // namespace optiling
