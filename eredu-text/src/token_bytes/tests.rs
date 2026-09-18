use super::*;
use tokenizers::decoders::{
    byte_fallback::ByteFallback, byte_level::ByteLevel, sequence::Sequence,
};

#[test]
fn exact_byte_worker_preserves_alphabet_fallback_and_atomic_refusal() {
    let decoder = DecoderWrapper::ByteLevel(ByteLevel::default());
    let encoding = TokenByteEncoding::inspect(Some(&decoder)).unwrap();
    let mut escaped = 256;
    for byte in 0..=255u32 {
        let scalar = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte
        } else {
            let scalar = escaped;
            escaped += 1;
            scalar
        };
        let mut utf8 = [0; 4];
        let token = char::from_u32(scalar).unwrap().encode_utf8(&mut utf8);
        let mut output = [0];
        assert_eq!(encoding.encoded_len(token).unwrap(), 1);
        encoding.write(token, &mut output).unwrap();
        assert_eq!(output, [byte as u8]);
    }
    let mut untouched = [23; 4];
    assert!(matches!(
        encoding.write("a🙂", &mut untouched),
        Err(TokenByteError::Destination)
    ));
    assert_eq!(untouched, [23; 4]);
    assert!(matches!(
        encoding.write("a", &mut untouched),
        Err(TokenByteError::Destination)
    ));
    assert_eq!(untouched, [23; 4]);
    for token in ["a🙂", "Ġ🙂", "\n{\"name\":\"reading\"}", "Ġa\n"] {
        let mut output = vec![0; encoding.encoded_len(token).unwrap()];
        encoding.write(token, &mut output).unwrap();
        assert_eq!(
            output,
            token.as_bytes(),
            "raw fallback covers the entire token"
        );
        let mut trie = vec![0; encoding.trie_token_len(token, false).unwrap()];
        encoding.write_trie_token(token, false, &mut trie).unwrap();
        assert_eq!(trie, output);
        let mut special = vec![0; encoding.trie_token_len(token, true).unwrap()];
        encoding
            .write_trie_token(token, true, &mut special)
            .unwrap();
        assert_eq!(special[0], 0xff);
        assert_eq!(&special[1..], token.as_bytes());
    }
    let decoder = DecoderWrapper::Sequence(Sequence::new(vec![
        DecoderWrapper::ByteLevel(ByteLevel::default()),
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Replace(tokenizers::normalizers::replace::Replace::new("▁", " ").unwrap()),
    ]));
    let encoding = TokenByteEncoding::inspect(Some(&decoder)).unwrap();
    for (token, expected) in [
        ("<0xFF>", vec![255]),
        ("<0x+a>", vec![10]),
        ("é▁🙂", "é 🙂".as_bytes().to_vec()),
    ] {
        let mut output = vec![0; encoding.encoded_len(token).unwrap()];
        encoding.write(token, &mut output).unwrap();
        assert_eq!(output, expected);
    }
    assert!(encoding.control_bytes().unwrap() > 0);
    assert!(matches!(
        encoding.write("<0xGG>", &mut untouched),
        Err(TokenByteError::Fallback(Some(_)))
    ));
    assert_eq!(untouched, [23; 4]);
}
