// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    const auto* weight = context->GetInputShape(1);
    auto* y = context->GetOutputShape(0);
    if (x == nullptr || weight == nullptr || y == nullptr ||
        x->GetDimNum() != 2 || weight->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    y->SetDimNum(2);
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, weight->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_FLOAT16);
}
IMPL_OP_INFERSHAPE(VllmQuantGroupedMatmulDequantV310)
    .InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
