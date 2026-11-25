// A 64-column by 64-row output tile. The packed loader expands only the
// current K panel into threadgroup memory. Steel consumes FP32 fragments of
// the same FP16-rounded operands as the smaller packed tile.
constexpr uint TM = 64;
constexpr uint TN = 64;
constexpr uint TK = 16;
constexpr uint WM = 2;
constexpr uint WN = 2;
constexpr uint THREADS = 32 * WM * WN;
constexpr uint PITCH = TK + 4;
constexpr uint BLOCKS = WIDTH / 256;
const uint rows = input_shape[0];
const uint first_col = threadgroup_position_in_grid.y * TN;
const uint first_row = threadgroup_position_in_grid.z * TM;
const uint live_rows = min(TM, rows - first_row);
const uint live_cols = min(TN, uint(OUTPUTS) - first_col);
const uint tid = thread_index_in_threadgroup;

threadgroup float act[TM * PITCH];
threadgroup float weights[TN * PITCH];
mlx::steel::BlockMMA<float, float, TM, TN, TK, WM, WN, false, true,
    PITCH, PITCH, float, mlx::steel::TransformNone<float, float>> mma(
        simdgroup_index_in_threadgroup, tid & 31);

for (uint k = 0; k < WIDTH; k += TK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TN) {
        uint row = first_col + min(tid, live_cols - 1);
        half values[16];
        qk_dequantize_16(weight, row * BLOCKS * QK_BLOCK_BYTES, k / TK, values);
        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(weights + tid * PITCH);
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            dst[j] = float4(values[4*j], values[4*j+1], values[4*j+2], values[4*j+3]);
        }
    }
    for (uint part = tid; part < TM * (TK/4); part += THREADS) {
        uint row = part / (TK/4);
        uint col = part % (TK/4);
        device const packed_float4* src = reinterpret_cast<device const packed_float4*>(
            input + (first_row + min(row, live_rows - 1)) * WIDTH + k + col * 4);
        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(act + row * PITCH);
        dst[col] = float4(half4(float4(*src)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(act, weights);
}

device float* result = out + ulong(first_row) * OUTPUTS + first_col;
if (live_rows == TM && live_cols == TN) {
    mma.store_result(result, OUTPUTS);
} else {
    mma.store_result_safe(result, OUTPUTS, short2(live_cols, live_rows));
}
