fn write_payload_tensors(
    destination: &std::path::Path,
    tensors: BTreeMap<String, NumericTensor>,
    integer_keys: &BTreeSet<&str>,
) -> BTreeMap<String, (Vec<i32>, Vec<u32>)> {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let expected = tensors
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                (
                    value.shape.clone(),
                    value.data.iter().map(|value| value.to_bits()).collect(),
                ),
            )
        })
        .collect();
    let encoded = tensors
        .into_iter()
        .map(|(name, value)| {
            let integer = integer_keys.contains(name.as_str());
            let dtype = if integer { Dtype::I32 } else { Dtype::F32 };
            let bytes = value
                .data
                .iter()
                .flat_map(|value| {
                    if integer {
                        (*value as i32).to_le_bytes()
                    } else {
                        value.to_le_bytes()
                    }
                })
                .collect::<Vec<_>>();
            (
                name,
                (
                    value
                        .shape
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    dtype,
                    bytes,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let views = encoded
        .iter()
        .map(|(name, (shape, dtype, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, destination).unwrap();
    expected
}
