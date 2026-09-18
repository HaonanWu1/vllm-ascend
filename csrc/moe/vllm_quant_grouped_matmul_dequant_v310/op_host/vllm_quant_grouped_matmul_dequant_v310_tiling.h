// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_QUANT_GROUPED_MATMUL_DEQUANT_V310_TILING_H
#define VLLM_QUANT_GROUPED_MATMUL_DEQUANT_V310_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
// Private ABI for this op only. Unused fields retain the shared kernel's
// interface; no CANN tiling library or built-in binary metadata is borrowed.
BEGIN_TILING_DATA_DEF(VllmQuantGroupedMatmulDequantV310TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, originM);
    TILING_DATA_FIELD_DEF(uint32_t, originN);
    TILING_DATA_FIELD_DEF(uint32_t, originK);
    TILING_DATA_FIELD_DEF(uint32_t, originE);
    TILING_DATA_FIELD_DEF(uint32_t, originKAligned32);
    TILING_DATA_FIELD_DEF(uint32_t, originKAligned512);
    TILING_DATA_FIELD_DEF(uint32_t, fracM);
    TILING_DATA_FIELD_DEF(uint32_t, fracN);
    TILING_DATA_FIELD_DEF(uint32_t, fracK);
    TILING_DATA_FIELD_DEF(uint32_t, tailM);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreFracN);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreFracNTail);
    TILING_DATA_FIELD_DEF(uint32_t, processXKBaseNMax);
    TILING_DATA_FIELD_DEF(uint32_t, perToken);
    TILING_DATA_FIELD_DEF(uint32_t, dynamicQuant);
    TILING_DATA_FIELD_DEF(uint32_t, smoothScale);
    TILING_DATA_FIELD_DEF(uint32_t, isXScaleHalf);
    TILING_DATA_FIELD_DEF(uint32_t, dynamicBaseK);
    TILING_DATA_FIELD_DEF(uint32_t, dynamicBaseKTail);
    TILING_DATA_FIELD_DEF(uint32_t, dynamicIterK);
    TILING_DATA_FIELD_DEF(uint32_t, MCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, NCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreMTail);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreNTail);
    TILING_DATA_FIELD_DEF(uint32_t, baseMNum);
    TILING_DATA_FIELD_DEF(uint32_t, baseNNum);
    TILING_DATA_FIELD_DEF(uint32_t, baseNNum_2);
    TILING_DATA_FIELD_DEF(uint32_t, baseKNum);
    TILING_DATA_FIELD_DEF(uint32_t, baseKNum_2);
    TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TILING_DATA_FIELD_DEF(uint32_t, baseK_2);
    TILING_DATA_FIELD_DEF(uint32_t, baseKTail);
    TILING_DATA_FIELD_DEF(uint32_t, baseKTail_2);
    TILING_DATA_FIELD_DEF(uint32_t, processXKloopPerfracM);
    TILING_DATA_FIELD_DEF(uint32_t, processXKloop);
    TILING_DATA_FIELD_DEF(uint32_t, processXKBaseN);
    TILING_DATA_FIELD_DEF(uint32_t, processXKTailN);
    TILING_DATA_FIELD_DEF(uint32_t, swiftGEMVThreshold);
    TILING_DATA_FIELD_DEF(uint32_t, baseFracK);
    TILING_DATA_FIELD_DEF(uint32_t, baseFracN);
    TILING_DATA_FIELD_DEF(uint32_t, baseFracNL0C);
    TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
    TILING_DATA_FIELD_DEF(uint32_t, ubBaseKTail);
    TILING_DATA_FIELD_DEF(uint32_t, ubIterK);
    TILING_DATA_FIELD_DEF(uint32_t, ubKMask);
    TILING_DATA_FIELD_DEF(uint64_t, workspaceOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(VllmQuantGroupedMatmulDequantV310, VllmQuantGroupedMatmulDequantV310TilingData)
}  // namespace optiling
#endif
