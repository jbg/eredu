use super::*;

thread_local! {
    static QUANTIZE: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static DOT: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

pub(super) fn linear(
    input: &Array,
    view: &storage::NativeQuantizedTensor,
    encoding: GgmlType,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = dispatch::validate_activation_dtype(input)?;
    let rows = i32::try_from(input.size() / view.columns as usize)
        .map_err(|_| Exception::custom("packed Metal activation row count overflows"))?;
    let mut shape = input.shape().to_vec();
    *shape.last_mut().expect("validated projection width") = view.rows;
    if rows == 0 {
        return safemlx::ops::zeros_dtype(&shape, dtype, stream);
    }
    let (width, bytes) = view.format().block_geometry();
    let blocks = view.columns / width;
    let count = rows
        .checked_mul(blocks)
        .ok_or_else(|| Exception::custom("packed Metal activation block count overflows"))?;
    let activation = CustomKernelConfig::new()
        .with_template_arg_int("WIDTH", width)
        .with_thread_group([64, 1, 1])
        .with_grid([count, 1, 1])
        .with_output_arg([count, width], Dtype::Int8)
        .with_output_arg([count], Dtype::Float32)
        .with_output_arg([count, if width == 32 { 1 } else { 8 }], Dtype::Int32);
    let inputs = QUANTIZE.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(activation_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("initialized activation kernel")
            .apply_device([input], &activation, stream)
    })?;
    let format = match encoding {
        GgmlType::Q5_0 => 0,
        GgmlType::Q8_0 => 1,
        GgmlType::Q4K => 2,
        GgmlType::Q5K => 3,
        GgmlType::Q6K => 4,
        _ => unreachable!("selected integer dot encoding"),
    };
    let config = CustomKernelConfig::new()
        .with_template_arg_int("FORMAT", format)
        .with_template_arg_int("BLOCKS", blocks)
        .with_template_arg_int("WIDTH", width)
        .with_template_arg_int("BLOCK_BYTES", bytes)
        .with_template_arg_int("OUTPUTS", view.rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("TILED_ROWS", rows / 4 * 4)
        .with_template_arg_dtype("T", dtype)
        .with_thread_group([64, 1, 1])
        .with_grid([view.rows, rows, 1])
        .with_output_arg(shape, dtype);
    DOT.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(dot_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("initialized packed dot kernel")
            .apply_one_device(
                [&inputs[0], &inputs[1], &inputs[2], view.storage.bytes()],
                &config,
                stream,
            )
    })
}

fn activation_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "gguf_integer_dot_activations",
        ["input"],
        ["codes", "scales", "sums"],
        r#"
        uint block = thread_position_in_grid.x;
        uint start = block * WIDTH;
        float peak = 0.0f;
        for (uint i = 0; i < WIDTH; ++i) {
            float value = float(input[start + i]);
            if (abs(value) > abs(peak)) peak = value;
        }
        if (WIDTH == 32) {
            float scale = metal::precise::divide(abs(peak), 127.0f);
            float inverse = scale == 0.0f ? 0.0f : metal::precise::divide(1.0f, scale);
            scales[block] = float(half(scale));
            for (uint i = 0; i < 32; ++i) {
                codes[start + i] = char(rint(float(input[start + i]) * inverse));
            }
            sums[block] = 0;
        } else {
            float inverse = peak == 0.0f ? 0.0f : metal::precise::divide(-127.0f, peak);
            scales[block] = peak == 0.0f ? 0.0f : metal::precise::divide(1.0f, inverse);
            for (uint group = 0; group < 8; ++group) {
                int sum = 0;
                for (uint j = 0; j < 32; ++j) {
                    uint i = start + group * 32 + j;
                    char code = char(rint(float(input[i]) * inverse));
                    codes[i] = code;
                    sum += int(code);
                }
                sums[block * 8 + group] = sum;
            }
        }
        "#,
        "",
        true,
        false,
    )
}

fn dot_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "gguf_integer_dot_projection",
        ["codes", "scales", "sums", "weights"],
        ["out"],
        r#"
        uint column = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint offset = row * BLOCKS;
        const device uchar* weights_row = weights + (ROW_START + column) * BLOCKS * BLOCK_BYTES;
        float sum = 0.0f;
        if (FORMAT == 0) {
            float4 even = 0.0f;
            float4 odd = 0.0f;
            uint b = 0;
            for (; b + 1 < BLOCKS; b += 2) {
                const device uchar* w = weights_row + b * BLOCK_BYTES;
                uint i = offset + b;
                even = fma(float4(q5_dots(codes + i * WIDTH, w)), scales[i] * packed_half(w), even);
                odd = fma(float4(q5_dots(codes + (i + 1) * WIDTH, w + BLOCK_BYTES)),
                          scales[i + 1] * packed_half(w + BLOCK_BYTES), odd);
            }
            sum = ((even.x + even.y) + (even.z + even.w)) +
                  ((odd.x + odd.y) + (odd.z + odd.w));
            if (b < BLOCKS) {
                const device uchar* w = weights_row + b * BLOCK_BYTES;
                uint i = offset + b;
                int4 dots = q5_dots(codes + i * WIDTH, w);
                sum = fma(float(dots.x + dots.y + dots.z + dots.w), scales[i] * packed_half(w), sum);
            }
        } else {
            for (uint b = 0; b < BLOCKS; ++b) {
                const device uchar* w = weights_row + b * BLOCK_BYTES;
                uint i = offset + b;
                const device int8_t* a = codes + i * WIDTH;
                if (FORMAT == 1) {
                    int dot = 0;
                    for (uint j = 0; j < 32; ++j) dot += int(a[j]) * int(as_type<char>(w[2 + j]));
                    sum = fma(float(dot), scales[i] * packed_half(w), sum);
                } else if (FORMAT == 4) {
                    sum = q6_k_block(a, scales[i], w, row < TILED_ROWS, sum);
                } else {
                    sum = min_k_block(a, scales[i], sums + i * 8, w, FORMAT == 3, sum);
                }
            }
        }
        out[row * OUTPUTS + column] = T(sum);
        "#,
        include_str!("packed_dot.metal"),
        true,
        false,
    )
}
