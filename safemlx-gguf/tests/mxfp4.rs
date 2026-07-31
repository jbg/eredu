use std::{collections::BTreeMap, io::Cursor};

use safemlx_gguf::{ConvertedTensor, Endian, GgmlType, Reader, TensorInput, Writer, WriterOptions};

fn fixture(endian: Endian) -> Vec<u8> {
    let mut block = [0u8; 17];
    block[0] = 130;
    for index in 0..16 {
        block[index + 1] = (index as u8) | ((15 - index as u8) << 4);
    }
    let mut output = Cursor::new(Vec::new());
    Writer::new(WriterOptions {
        version: 3,
        endian,
        alignment: 32,
    })
    .unwrap()
    .write(
        &mut output,
        &BTreeMap::new(),
        &[TensorInput {
            name: "experts.weight",
            dimensions: &[32],
            ggml_type: GgmlType::MxFp4,
            data: &block,
        }],
    )
    .unwrap();
    output.into_inner()
}

#[test]
fn type_39_decodes_to_mlx_mxfp4_weights_and_e8m0_scales() {
    assert_eq!(GgmlType::from_code(39), GgmlType::MxFp4);
    assert_eq!(GgmlType::MxFp4.block_and_bytes().unwrap(), (32, 17));
    for endian in [Endian::Little, Endian::Big] {
        let bytes = fixture(endian);
        let mut reader = Reader::new(Cursor::new(bytes.clone())).unwrap();
        let descriptor = reader.tensors()[0].clone();
        assert_eq!(
            reader.read_raw(&descriptor).unwrap(),
            [
                130, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c,
                0x2d, 0x1e, 0x0f
            ]
        );
        let ConvertedTensor::MxFp4(tensor) = reader.read_tensor(&descriptor).unwrap() else {
            panic!("type 39 did not produce MXFP4");
        };
        assert_eq!(tensor.weight_shape, [4]);
        assert_eq!(tensor.scale_shape, [1]);
        assert_eq!(tensor.scales, [130]);
        assert_eq!(
            tensor.weights,
            [0x7654_3210, 0xfedc_ba98, 0x89ab_cdef, 0x0123_4567]
        );
    }
}

#[test]
fn writer_preserves_raw_type_39_blocks() {
    let bytes = fixture(Endian::Little);
    let mut reader = Reader::new(Cursor::new(bytes)).unwrap();
    let descriptor = reader.tensors()[0].clone();
    let raw = reader.read_raw(&descriptor).unwrap();
    assert_eq!(raw.len(), 17);
    assert_eq!(raw[0], 130);
}
