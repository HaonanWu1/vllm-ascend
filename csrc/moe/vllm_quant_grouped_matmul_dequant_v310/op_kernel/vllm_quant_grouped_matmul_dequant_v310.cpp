// SPDX-License-Identifier: Apache-2.0
// Reuse the validated arithmetic; register a distinct kernel and tiling ABI.
#include "kernel_operator.h"
using QuantMatmulDequantTilingData = VllmQuantGroupedMatmulDequantV310TilingData;
#include "quant_matmul_dequant_grouped.h"

extern "C" __global__ __aicore__ void vllm_quant_grouped_matmul_dequant_v310(
    GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
    GM_ADDR x_scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    SetAtomicNone();
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        QuantMatmulDequantGrouped op;
        op.Init(x, quantized_weight, weight_scale, group_list, nullptr, x_scale,
                nullptr, nullptr, y, workspace + tilingData.workspaceOffset, &tilingData);
        op.Process();
    }
    SetMaskNorm();
    ResetMask();
}
