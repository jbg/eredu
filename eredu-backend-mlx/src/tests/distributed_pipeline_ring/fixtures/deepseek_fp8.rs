// Published block-FP8 encodings through the ordinary component/prediction driver.
fn write_deepseek_v4_fp8_fixture(directory: &Path, dspark: bool, ue8m0: bool) {
    write_deepseek_v4_fp8_fixture_kind(directory, dspark, ue8m0, false);
}

fn write_deepseek_v4_fp8_fixture_kind(directory: &Path, dspark: bool, ue8m0: bool, mixed: bool) {
    write_deepseek_v4_fixture_kind(directory, if dspark { 2 } else { 1 }, dspark, true, true);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    // Complete 128-wide blocks on both TP owners, including grouped attention
    // factors, compressed attention and target/prediction expert projections.
    for (name, value) in [
        ("hidden_size", 256),
        ("moe_intermediate_size", 256),
        ("vocab_size", 256),
        ("head_dim", 128),
        ("qk_rope_head_dim", 64),
        ("q_lora_rank", 128),
        ("o_lora_rank", 128),
        ("index_head_dim", 128),
    ] {
        config[name] = value.into();
    }
    if dspark {
        config["dspark_markov_rank"] = 256.into();
    }
    config["expert_dtype"] = if mixed { "fp4" } else { "fp8" }.into();
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128]
    });
    if ue8m0 {
        config["quantization_config"]["scale_fmt"] = "ue8m0".into();
    }
    let args = eredu_architectures::deepseek::parse_v4_config(&config).unwrap();
    let plan = eredu_architectures::deepseek::v4_safetensors_plan(&args).unwrap();
    use eredu_checkpoint::{
        schema::{StoredDtypeConstraint, TensorRole},
        StoredDtype,
    };
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let name = &tensor.key;
            let count = tensor.shape.iter().product::<usize>();
            let (dtype, bytes): (Dtype, Vec<u8>) = match tensor.dtype {
                StoredDtypeConstraint::Exact(StoredDtype::U32) => (
                    Dtype::U32,
                    (0..count)
                        .flat_map(|i| {
                            let packed = (0..8).fold(0_u32, |word, nibble| {
                                let code =
                                    ((k2_fp8_phase(name) + 3 * (8 * i + nibble)) % 15 + 1) as u32;
                                word | (code << (4 * nibble))
                            });
                            packed.to_le_bytes()
                        })
                        .collect(),
                ),
                _ if mixed && tensor.role == TensorRole::Companion && name.ends_with(".scale") => (
                    Dtype::U8,
                    (0..count)
                        .map(|i| 120 + ((k2_fp8_phase(name) + 3 * i) % 4) as u8)
                        .collect(),
                ),
                StoredDtypeConstraint::Exact(StoredDtype::F8E4M3) => (
                    Dtype::F8_E4M3,
                    (0..count).map(|i| k2_fp8_code(name, i)).collect(),
                ),
                StoredDtypeConstraint::Exact(StoredDtype::F8E8M0) => (
                    Dtype::F8_E8M0,
                    (0..count)
                        .map(|i| 120 + ((k2_fp8_phase(name) + 3 * i) % 4) as u8)
                        .collect(),
                ),
                StoredDtypeConstraint::Exact(StoredDtype::I32) => (
                    Dtype::I32,
                    (0..count)
                        .flat_map(|i| ((i % 4) as i32).to_le_bytes())
                        .collect(),
                ),
                _ => (
                    Dtype::F32,
                    (0..count)
                        .flat_map(|i| {
                            let value = if tensor.role == TensorRole::Companion {
                                k2_fp8_scale(name.strip_suffix("_scale_inv").unwrap(), i)
                            } else if name.ends_with("norm.weight") {
                                0.95 + 0.02 * (i % 7) as f32
                            } else {
                                ((i * 37 + k2_fp8_phase(name)) % 101) as f32 * 0.001 - 0.05
                            };
                            value.to_le_bytes()
                        })
                        .collect(),
                ),
            };
            (name.clone(), tensor.shape.clone(), dtype, bytes)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join("component-v4-fp8-fixture.json"),
        serde_json::to_vec(&serde_json::json!({"scale": if ue8m0 { "ue8m0" } else { "f32" }}))
            .unwrap(),
    )
    .unwrap();
    if mixed {
        std::fs::write(directory.join("component-v4-mixed-fp8-fixture.json"), b"{}").unwrap();
    }
}

fn verify_v4_fp8_source_row(
    source: &safetensors::SafeTensors<'_>,
    parameter: &str,
    shape: &[u64],
    region: &eredu_core::parameters::ParameterRegion,
    values: &[f32],
) {
    for (index, actual) in values.iter().enumerate() {
        let mut remaining = index as u64;
        let mut coordinates = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            coordinates[axis] = region.starts[axis] + remaining % region.shape[axis];
            remaining /= region.shape[axis];
        }
        let (name, coordinates) =
            if let Some(root) = parameter.strip_suffix(".switch_mlp.gate_up_proj") {
                let half = shape[1] / 2;
                let field = if coordinates[1] < half { "w1" } else { "w3" };
                (
                    format!("{root}.experts.{}.{field}.weight", coordinates[0]),
                    vec![coordinates[1] % half, coordinates[2]],
                )
            } else if let Some(root) = parameter.strip_suffix(".switch_mlp.down_proj") {
                (
                    format!("{root}.experts.{}.w2.weight", coordinates[0]),
                    vec![coordinates[1], coordinates[2]],
                )
            } else if let Some(root) = parameter.strip_suffix(".experts.gate_up_proj") {
                let half = shape[1] / 2;
                let field = if coordinates[1] < half {
                    "gate_proj"
                } else {
                    "up_proj"
                };
                (
                    format!("{root}.experts.{}.{field}.weight", coordinates[0]),
                    vec![coordinates[1] % half, coordinates[2]],
                )
            } else if let Some(root) = parameter.strip_suffix(".experts.down_proj") {
                (
                    format!("{root}.experts.{}.down_proj.weight", coordinates[0]),
                    vec![coordinates[1], coordinates[2]],
                )
            } else {
                (parameter.to_owned(), coordinates)
            };
        let weight = source
            .tensor(&name)
            .unwrap_or_else(|error| panic!("{parameter}: {error}"));
        assert_eq!(coordinates.len(), weight.shape().len(), "{name}");
        assert_eq!(coordinates.len(), 2, "source matrix {name}");
        let row = coordinates[0] as usize;
        let column = coordinates[1] as usize;
        if weight.dtype() == Dtype::U32 {
            let logical_width = weight.shape()[1] * 8;
            let byte = weight.data()[row * logical_width / 2 + column / 2];
            let code = (byte >> (4 * (column % 2))) & 15;
            let magnitude = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][usize::from(code & 7)];
            let scale = source
                .tensor(&format!("{}.scale", name.strip_suffix(".weight").unwrap()))
                .unwrap();
            assert_eq!(scale.dtype(), Dtype::U8);
            let exponent = scale.data()[row * logical_width / 32 + column / 32];
            let expected = magnitude
                * if code & 8 == 0 { 1. } else { -1. }
                * 2_f32.powi(i32::from(exponent) - 127);
            assert_eq!(
                *actual, expected,
                "independent MXFP4 source {name} row={row} column={column}"
            );
            continue;
        }
        assert_eq!(weight.dtype(), Dtype::F8_E4M3, "{name}");
        let code = weight.data()[row * weight.shape()[1] + column];
        let exponent = ((code >> 3) & 15) as i32;
        let mantissa = (code & 7) as f32;
        let scalar = if exponent == 0 {
            mantissa * 2_f32.powi(-9)
        } else {
            (1. + mantissa / 8.) * 2_f32.powi(exponent - 7)
        };
        let scale = source.tensor(&format!("{}_scale_inv", name)).unwrap();
        let block = row / 128 * weight.shape()[1].div_ceil(128) + column / 128;
        let scale = match scale.dtype() {
            Dtype::F8_E8M0 => 2_f32.powi(i32::from(scale.data()[block]) - 127),
            Dtype::F32 => {
                f32::from_le_bytes(scale.data()[block * 4..block * 4 + 4].try_into().unwrap())
            }
            dtype => panic!("unexpected fixture scale {dtype:?}"),
        };
        assert_eq!(
            *actual,
            scalar * if code & 128 == 0 { 1. } else { -1. } * scale,
            "independent V4 FP8 source {name} row={row} column={column}"
        );
    }
}
