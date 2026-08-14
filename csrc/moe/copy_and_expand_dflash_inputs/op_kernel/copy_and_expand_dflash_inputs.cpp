#include "kernel_operator.h"

using namespace AscendC;

// ONE_BLK_SIZE comes from AscendC namespace (32 bytes per block)

// AscendC replacement for the Triton kernel
// ``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid``.
//
// It builds the parallel-drafting draft inputs for DFlash / DSpark:
//   1. Copies target/context positions and rebuilds their slot mapping from
//      request boundaries and the physical draft-cache block layout.
//   2. For every request builds the query block of
//      ``num_query_per_req`` tokens: a bonus token followed by
//      ``parallel_drafting_token_id`` mask tokens, with their positions
//      and KV-cache slot mapping (looked up via the block table).
//   3. Emits the ``token_indices_to_sample`` used by the sampler.
//
// The op runs on a single core (usedCoreNum == 1).
//
// 310P notes (__CCE_AICORE__ == 200):
//   * DataCopyPad is NOT available, so every bulk GM transfer uses the
//     plain block-oriented DataCopy, which requires a 32B aligned GM
//     start and a 32B-multiple length.
//   * All scalar metadata (query_start_loc, seq_lens, next_token_ids,
//     num_rejected_tokens, block_table entries, the last context
//     position) are read directly from GM via GlobalTensor::GetValue,
//     exactly as the proven 310P causal_conv1d_v310 kernel does. This
//     avoids out-of-bounds rounded UB reads on exactly-sized GM tensors
//     and the associated MTE2->S synchronization.
//   * The per-request query outputs are assembled in UB and each output
//     array is flushed to GM once from offset 0 (32B aligned); the
//     block-rounded tail lands in the destination tensor's own
//     allocation padding.
//   * Context positions and generated slots are tiled on CTX_TILE (a multiple
//     of 8), so every tile's GM offset stays 32B aligned.
class CopyAndExpandDflashInputsKernel {
public:
    __aicore__ inline CopyAndExpandDflashInputsKernel() {}

    __aicore__ inline void Init(GM_ADDR nextTokenIds, GM_ADDR targetPositions,
                                GM_ADDR contextSlotMapping, GM_ADDR queryStartLoc,
                                GM_ADDR seqLens, GM_ADDR blockTable,
                                GM_ADDR numRejectedTokens,
                                GM_ADDR outInputIds, GM_ADDR outQueryPositions,
                                GM_ADDR outQuerySlotMapping, GM_ADDR outContextPositions,
                                GM_ADDR outContextSlotMapping, GM_ADDR outTokenIndices,
                                const CopyAndExpandDflashInputsTilingData* tilingData)
    {
        numReqs = tilingData->numReqs;
        numContext = tilingData->numContext;
        blockTableStride = tilingData->blockTableStride;
        parallelDraftingTokenId = tilingData->parallelDraftingTokenId;
        blockSize = tilingData->blockSize;
        numQueryPerReq = tilingData->numQueryPerReq;
        numSpeculativeTokens = tilingData->numSpeculativeTokens;
        sampleFromAnchor = tilingData->sampleFromAnchor;

        numQueryTotal = numReqs * numQueryPerReq;
        numTokenIndicesTotal = numReqs * numSpeculativeTokens;

        gmNextTokenIds.SetGlobalBuffer((__gm__ int32_t*)nextTokenIds, numReqs);
        gmTargetPositions.SetGlobalBuffer((__gm__ int32_t*)targetPositions, numContext);
        gmContextSlotMapping.SetGlobalBuffer((__gm__ int32_t*)contextSlotMapping, numContext);
        gmQueryStartLoc.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, numReqs + 1);
        gmSeqLens.SetGlobalBuffer((__gm__ int32_t*)seqLens, numReqs);
        gmBlockTable.SetGlobalBuffer((__gm__ int32_t*)blockTable, blockTableStride * numReqs);
        gmNumRejectedTokens.SetGlobalBuffer((__gm__ int32_t*)numRejectedTokens, numReqs);

        gmOutInputIds.SetGlobalBuffer((__gm__ int32_t*)outInputIds, numQueryTotal);
        gmOutQueryPositions.SetGlobalBuffer((__gm__ int32_t*)outQueryPositions, numQueryTotal);
        gmOutQuerySlotMapping.SetGlobalBuffer((__gm__ int32_t*)outQuerySlotMapping, numQueryTotal);
        gmOutContextPositions.SetGlobalBuffer((__gm__ int32_t*)outContextPositions, numContext);
        gmOutContextSlotMapping.SetGlobalBuffer((__gm__ int32_t*)outContextSlotMapping, numContext);
        gmOutTokenIndices.SetGlobalBuffer((__gm__ int32_t*)outTokenIndices, numTokenIndicesTotal);

        // Full query output arrays, assembled in UB and written back once.
        pipe.InitBuffer(qIdsBuf, AlignUpBytes(numQueryTotal * sizeof(int32_t)));
        pipe.InitBuffer(qPosBuf, AlignUpBytes(numQueryTotal * sizeof(int32_t)));
        pipe.InitBuffer(qSlotBuf, AlignUpBytes(numQueryTotal * sizeof(int32_t)));
        pipe.InitBuffer(tokBuf, AlignUpBytes(numTokenIndicesTotal * sizeof(int32_t)));

        // Tile buffers for context positions and generated physical slots.
        pipe.InitBuffer(ctxPosBuf, AlignUpBytes(CTX_TILE * sizeof(int32_t)));
        pipe.InitBuffer(ctxSlotBuf, AlignUpBytes(CTX_TILE * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        CopyContext();
        if (numReqs == 0) return;

        LocalTensor<int32_t> lIds = qIdsBuf.Get<int32_t>();
        LocalTensor<int32_t> lPos = qPosBuf.Get<int32_t>();
        LocalTensor<int32_t> lSlot = qSlotBuf.Get<int32_t>();
        LocalTensor<int32_t> lTok = tokBuf.Get<int32_t>();

        for (uint32_t r = 0; r < numReqs; r++) {
            ProcessOneRequest(r, lIds, lPos, lSlot, lTok);
        }

        // All query arrays assembled by scalar writes; flush to GM once.
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopy(gmOutInputIds, lIds, AlignUpElems((int32_t)numQueryTotal));
        DataCopy(gmOutQueryPositions, lPos, AlignUpElems((int32_t)numQueryTotal));
        DataCopy(gmOutQuerySlotMapping, lSlot, AlignUpElems((int32_t)numQueryTotal));
        if (numTokenIndicesTotal > 0) {
            DataCopy(gmOutTokenIndices, lTok, AlignUpElems((int32_t)numTokenIndicesTotal));
        }
    }

private:
    static constexpr int32_t CTX_TILE = 4096;
    static constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int32_t);  // 8

    // Round a byte count up to a 32B block, with a one-block floor so a
    // degenerate empty input never requests a zero-size UB allocation.
    static __aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
    {
        uint32_t aligned = (bytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        return aligned < ONE_BLK_SIZE ? ONE_BLK_SIZE : aligned;
    }

    // Round an int32 element count up to a 32B (8-element) block. GM
    // allocations are block-padded on NPU so a rounded DataCopy stays
    // in-bounds of the target tensor's own allocation.
    static __aicore__ inline int32_t AlignUpElems(int32_t count)
    {
        return (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
    }

    // Copy context positions and generate physical cache slots. Both buffers
    // are ordered across MTE2, scalar, and MTE3 pipes. Tile offsets are
    // multiples of CTX_TILE, i.e. 32B aligned.
    __aicore__ inline void CopyContext()
    {
        if (numContext == 0) return;
        LocalTensor<int32_t> posTile = ctxPosBuf.Get<int32_t>();
        LocalTensor<int32_t> slotTile = ctxSlotBuf.Get<int32_t>();
        uint32_t req = 0;
        int32_t ctxStart = numReqs > 0 ? gmQueryStartLoc.GetValue(0) : 0;
        int32_t ctxEnd = numReqs > 0 ? gmQueryStartLoc.GetValue(1) : 0;

        // Prime "buffer free to write" so the first read (MTE2) proceeds.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        for (int32_t off = 0; off < (int32_t)numContext; off += CTX_TILE) {
            int32_t count = (int32_t)numContext - off;
            if (count > CTX_TILE) count = CTX_TILE;
            int32_t aligned = AlignUpElems(count);

            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
            DataCopy(posTile, gmTargetPositions[off], aligned);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID4);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID4);

            for (int32_t j = 0; j < count; j++) {
                int32_t contextIndex = off + j;
                while (req < numReqs && contextIndex >= ctxEnd) {
                    req++;
                    if (req < numReqs) {
                        ctxStart = gmQueryStartLoc.GetValue(req);
                        ctxEnd = gmQueryStartLoc.GetValue(req + 1);
                    }
                }

                int32_t slot = PADDING_SLOT_ID;
                int32_t position = posTile.GetValue(j);
                if (req < numReqs && contextIndex >= ctxStart && contextIndex < ctxEnd) {
                    int32_t blockNum = position / (int32_t)blockSize;
                    int32_t blockId = gmBlockTable.GetValue((int32_t)(req * blockTableStride) + blockNum);
                    int64_t physicalSlot = (int64_t)blockId * (int64_t)blockSize +
                                           (int64_t)(position % (int32_t)blockSize);
                    slot = (int32_t)physicalSlot;
                }
                slotTile.SetValue(j, slot);
            }
            for (int32_t j = count; j < aligned; j++) {
                slotTile.SetValue(j, PADDING_SLOT_ID);
            }

            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID5);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID6);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID6);
            DataCopy(gmOutContextPositions[off], posTile, aligned);
            DataCopy(gmOutContextSlotMapping[off], slotTile, aligned);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }

    __aicore__ inline void ProcessOneRequest(uint32_t r,
                                             LocalTensor<int32_t>& lIds,
                                             LocalTensor<int32_t>& lPos,
                                             LocalTensor<int32_t>& lSlot,
                                             LocalTensor<int32_t>& lTok)
    {
        // Scalar metadata read directly from GM (proven 310P pattern).
        int32_t ctxStart = gmQueryStartLoc.GetValue(r);
        int32_t ctxEnd = gmQueryStartLoc.GetValue(r + 1);
        bool paddedRequest = ctxEnd <= ctxStart;

        // Clamp numRejected to [0, ctxLen] so validCtxEnd stays inside this
        // request's own context span: a negative value or one exceeding the
        // context length would otherwise index into the previous request
        // (data corruption) or produce validCtxEnd == 0 (see lastPos below).
        int32_t numRejected = gmNumRejectedTokens.GetValue(r);
        if (numRejected < 0) numRejected = 0;
        int32_t ctxLen = paddedRequest ? 0 : ctxEnd - ctxStart;
        if (numRejected > ctxLen) numRejected = ctxLen;
        int32_t validCtxEnd = ctxEnd - numRejected;

        int32_t seqLen = gmSeqLens.GetValue(r);
        int32_t effectiveSeqLen = seqLen - numRejected;
        if (effectiveSeqLen < 0) effectiveSeqLen = 0;
        int32_t nextTokenId = gmNextTokenIds.GetValue(r);
        // Guard the GM read: validCtxEnd == 0 (whole context rejected) would
        // otherwise read target_positions[-1] out of bounds.
        int32_t lastPos = (validCtxEnd > ctxStart) ? gmTargetPositions.GetValue(validCtxEnd - 1) : -1;

        int32_t queryBase = (int32_t)(r * numQueryPerReq);

        for (int32_t q = 0; q < (int32_t)numQueryPerReq; q++) {
            int32_t queryPos = lastPos + 1 + q;
            lPos.SetValue(queryBase + q, queryPos);

            int32_t queryCachePos = effectiveSeqLen + q;
            int64_t slot = PADDING_SLOT_ID;
            if (!paddedRequest) {
                int32_t blockNum = queryCachePos / (int32_t)blockSize;
                int64_t blockId =
                    (int64_t)gmBlockTable.GetValue((int32_t)(r * blockTableStride) + blockNum);
                slot = blockId * (int64_t)blockSize +
                       (int64_t)(queryCachePos % (int32_t)blockSize);
            }
            lSlot.SetValue(queryBase + q, (int32_t)slot);

            if (q == 0) {
                lIds.SetValue(queryBase + q, nextTokenId);
            } else {
                lIds.SetValue(queryBase + q, parallelDraftingTokenId);
            }
        }

        // token_indices_to_sample
        int32_t tokBase = (int32_t)(r * numSpeculativeTokens);
        if (sampleFromAnchor != 0) {
            for (int32_t q = 0; q < (int32_t)numSpeculativeTokens; q++) {
                lTok.SetValue(tokBase + q, queryBase + q);
            }
        } else {
            for (int32_t q = 1; q < (int32_t)numQueryPerReq; q++) {
                lTok.SetValue(tokBase + (q - 1), queryBase + q);
            }
        }
    }

private:
    static constexpr int32_t PADDING_SLOT_ID = -1;

    GlobalTensor<int32_t> gmNextTokenIds, gmTargetPositions, gmContextSlotMapping;
    GlobalTensor<int32_t> gmQueryStartLoc, gmSeqLens, gmBlockTable, gmNumRejectedTokens;
    GlobalTensor<int32_t> gmOutInputIds, gmOutQueryPositions, gmOutQuerySlotMapping;
    GlobalTensor<int32_t> gmOutContextPositions, gmOutContextSlotMapping, gmOutTokenIndices;

    uint32_t numReqs, numContext, blockTableStride;
    int32_t parallelDraftingTokenId;
    uint32_t blockSize, numQueryPerReq, numSpeculativeTokens, sampleFromAnchor;
    uint32_t numQueryTotal, numTokenIndicesTotal;

    TPipe pipe;
    TBuf<QuePosition::VECCALC> qIdsBuf, qPosBuf, qSlotBuf, tokBuf, ctxPosBuf, ctxSlotBuf;
};

extern "C" __global__ __aicore__ void copy_and_expand_dflash_inputs(
    GM_ADDR nextTokenIds, GM_ADDR targetPositions,
    GM_ADDR contextSlotMapping, GM_ADDR queryStartLoc,
    GM_ADDR seqLens, GM_ADDR blockTable,
    GM_ADDR numRejectedTokens,
    GM_ADDR outInputIds, GM_ADDR outQueryPositions,
    GM_ADDR outQuerySlotMapping, GM_ADDR outContextPositions,
    GM_ADDR outContextSlotMapping, GM_ADDR outTokenIndices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        CopyAndExpandDflashInputsKernel op;
        op.Init(nextTokenIds, targetPositions, contextSlotMapping, queryStartLoc,
                seqLens, blockTable, numRejectedTokens,
                outInputIds, outQueryPositions, outQuerySlotMapping,
                outContextPositions, outContextSlotMapping, outTokenIndices,
                &tilingData);
        op.Process();
    }
}
