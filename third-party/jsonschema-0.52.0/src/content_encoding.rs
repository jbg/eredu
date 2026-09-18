// `pub` only for the `__private` re-export consumed by generated code.
#![allow(clippy::must_use_candidate, clippy::missing_errors_doc)]

use crate::{
    error::ValidationError,
    validator::{
        workspace::{Component, Error},
        ValidationContext,
    },
};
use data_encoding::{Encoding, BASE32, BASE32HEX, BASE64, BASE64URL, HEXUPPER};
use serde_json::allocation::{Allocation, AllocationError, Allocator, Unenforced};

pub(crate) type ContentEncodingCheckType = fn(&str) -> bool;
pub(crate) type ContentEncodingConverterType =
    fn(&str) -> Result<Option<String>, ValidationError<'static>>;

#[derive(Clone, Copy)]
pub(crate) enum BuiltinEncoding {
    Base64,
    Base64Url,
    Base32,
    Base32Hex,
    Base16,
}
impl BuiltinEncoding {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "base64" => Some(Self::Base64),
            "base64url" => Some(Self::Base64Url),
            "base32" => Some(Self::Base32),
            "base32hex" => Some(Self::Base32Hex),
            "base16" => Some(Self::Base16),
            _ => None,
        }
    }
    fn encoding(self) -> Encoding {
        match self {
            Self::Base64 => BASE64,
            Self::Base64Url => BASE64URL,
            Self::Base32 => BASE32,
            Self::Base32Hex => BASE32HEX,
            Self::Base16 => HEXUPPER,
        }
    }
    fn decode(
        self,
        input: &str,
        allocation: &dyn Allocation,
    ) -> Result<Option<Vec<u8>>, AllocationError> {
        allocation.reserve(std::mem::size_of::<(
            Self,
            &str,
            Encoding,
            Option<Vec<u8>>,
            Vec<u8>,
            usize,
        )>())?;
        let encoding = self.encoding();
        let result = decode(&encoding, input.as_bytes(), allocation)?;
        if result.is_some() || !matches!(self, Self::Base16) {
            return Ok(result);
        }
        // Preserve the original Unicode uppercase retry for base16; uppercase
        // maps each scalar independently, using the same Rust Unicode tables.
        let size = input
            .chars()
            .flat_map(char::to_uppercase)
            .try_fold(0usize, |size, value| size.checked_add(value.len_utf8()))
            .ok_or(AllocationError::SizeOverflow)?;
        let mut uppercase = Vec::new();
        Allocator::new(allocation).grow(&mut uppercase, size)?;
        for character in input.chars().flat_map(char::to_uppercase) {
            uppercase.extend_from_slice(character.encode_utf8(&mut [0; 4]).as_bytes());
        }
        decode(&encoding, &uppercase, allocation)
    }
    fn convert(
        self,
        input: &str,
        allocation: &dyn Allocation,
    ) -> Result<Result<Option<String>, std::string::FromUtf8Error>, AllocationError> {
        Ok(match self.decode(input, allocation)? {
            Some(bytes) => String::from_utf8(bytes).map(Some),
            None => Ok(None),
        })
    }
    fn functions(self) -> (ContentEncodingCheckType, ContentEncodingConverterType) {
        match self {
            Self::Base64 => (is_base64, from_base64),
            Self::Base64Url => (is_base64url, from_base64url),
            Self::Base32 => (is_base32, from_base32),
            Self::Base32Hex => (is_base32hex, from_base32hex),
            Self::Base16 => (is_base16, from_base16),
        }
    }
}

// This is data-encoding's ordinary decode worker with its caller-owned output
// admitted before initialization. decode_len/decode_mut allocate no destination.
fn decode(
    encoding: &Encoding,
    input: &[u8],
    allocation: &dyn Allocation,
) -> Result<Option<Vec<u8>>, AllocationError> {
    let Ok(length) = encoding.decode_len(input.len()) else {
        return Ok(None);
    };
    let mut output = Vec::new();
    Allocator::new(allocation).grow(&mut output, length)?;
    output.resize(length, 0);
    let Ok(length) = encoding.decode_mut(input, &mut output) else {
        return Ok(None);
    };
    output.truncate(length);
    Ok(Some(output))
}

macro_rules! encoding_functions {
    ($($kind:ident, $check:ident, $convert:ident;)*) => {$ (
        pub fn $check(input: &str) -> bool { BuiltinEncoding::$kind.decode(input, &Unenforced).expect("ordinary content decode allocation").is_some() }
        pub fn $convert(input: &str) -> Result<Option<String>, ValidationError<'static>> {
            BuiltinEncoding::$kind.convert(input, &Unenforced).expect("ordinary content decode allocation").map_err(Into::into)
        }
    )*};
}
encoding_functions! { Base64, is_base64, from_base64; Base64Url, is_base64url, from_base64url; Base32, is_base32, from_base32; Base32Hex, is_base32hex, from_base32hex; Base16, is_base16, from_base16; }

pub(crate) fn default_content_encoding(
    name: &str,
) -> Option<(ContentEncodingCheckType, ContentEncodingConverterType)> {
    BuiltinEncoding::from_name(name).map(BuiltinEncoding::functions)
}

#[derive(Clone, Copy)]
pub(crate) enum ContentEncodingSource {
    Builtin(BuiltinEncoding),
    Custom {
        check: ContentEncodingCheckType,
        convert: ContentEncodingConverterType,
    },
}
pub(crate) enum ConversionError {
    Utf8(std::string::FromUtf8Error),
    Schema(ValidationError<'static>),
}
impl ContentEncodingSource {
    pub(crate) fn qualify(self) -> Result<(), Error> {
        match self {
            Self::Builtin(_) => Ok(()),
            Self::Custom { .. } => Err(Error::Unqualified(Component::Source(
                "custom content encoding",
            ))),
        }
    }
    pub(crate) fn check(self, input: &str, context: &mut ValidationContext) -> bool {
        match self {
            Self::Builtin(encoding) => context
                .workspace
                .with_json_allocations(|allocation| {
                    Ok(encoding.decode(input, allocation)?.is_some())
                })
                .unwrap_or(false),
            Self::Custom { check, .. } => {
                if context.workspace.original() {
                    return context.workspace.refuse(self.qualify().unwrap_err());
                }
                check(input)
            }
        }
    }
    pub(crate) fn convert(
        self,
        input: &str,
        context: &mut ValidationContext,
    ) -> Option<Result<Option<String>, ConversionError>> {
        match self {
            Self::Builtin(encoding) => context.workspace.with_json_allocations(|allocation| {
                Ok(encoding
                    .convert(input, allocation)?
                    .map_err(ConversionError::Utf8))
            }),
            Self::Custom { convert, .. } => {
                if context.workspace.original() {
                    context.workspace.refuse(self.qualify().unwrap_err());
                    return None;
                }
                Some(convert(input).map_err(ConversionError::Schema))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    // Test string: "foobar"
    const TEST_STRING: &str = "foobar";
    const TEST_BASE64: &str = "Zm9vYmFy";
    const TEST_BASE64URL: &str = "Zm9vYmFy"; // same as base64 for "foobar" (no +/)
    const TEST_BASE32: &str = "MZXW6YTBOI======";
    const TEST_BASE32HEX: &str = "CPNMUOJ1E8======";
    const TEST_BASE16_UPPER: &str = "666F6F626172";
    const TEST_BASE16_LOWER: &str = "666f6f626172";
    const TEST_BASE16_MIXED: &str = "666F6f626172";

    #[test_case(TEST_BASE64, true ; "valid base64")]
    #[test_case("not valid base64!!!", false ; "invalid base64 with special chars")]
    #[test_case("Zm9v====", false ; "invalid base64 padding")]
    fn test_is_base64(input: &str, expected: bool) {
        assert_eq!(is_base64(input), expected);
    }

    #[test_case(TEST_BASE64, Some(TEST_STRING) ; "decode valid base64")]
    #[test_case("invalid!", None ; "decode invalid base64")]
    fn test_from_base64(input: &str, expected: Option<&str>) {
        assert_eq!(
            from_base64(input).unwrap(),
            expected.map(std::string::ToString::to_string)
        );
    }

    #[test_case(TEST_BASE64URL, true ; "valid base64url")]
    #[test_case("PDw_Pz4-", true ; "base64url with url safe chars")]
    #[test_case("Zm9v+YmFy", false ; "base64 plus char invalid in base64url")]
    #[test_case("Zm9v/YmFy", false ; "base64 slash char invalid in base64url")]
    fn test_is_base64url(input: &str, expected: bool) {
        assert_eq!(is_base64url(input), expected);
    }

    #[test_case(TEST_BASE64URL, Some(TEST_STRING) ; "decode valid base64url")]
    #[test_case("invalid!", None ; "decode invalid base64url")]
    fn test_from_base64url(input: &str, expected: Option<&str>) {
        assert_eq!(
            from_base64url(input).unwrap(),
            expected.map(std::string::ToString::to_string)
        );
    }

    #[test_case(TEST_BASE32, true ; "valid base32")]
    #[test_case("not valid", false ; "invalid base32 text")]
    #[test_case("189", false ; "base32 invalid chars 1,8,9")]
    fn test_is_base32(input: &str, expected: bool) {
        assert_eq!(is_base32(input), expected);
    }

    #[test_case(TEST_BASE32, Some(TEST_STRING) ; "decode valid base32")]
    #[test_case("189!!!", None ; "decode invalid base32")]
    fn test_from_base32(input: &str, expected: Option<&str>) {
        assert_eq!(
            from_base32(input).unwrap(),
            expected.map(std::string::ToString::to_string)
        );
    }

    #[test_case(TEST_BASE32HEX, true ; "valid base32hex")]
    #[test_case("not valid", false ; "invalid base32hex text")]
    #[test_case("XYZ", false ; "base32hex invalid chars X,Y,Z")]
    fn test_is_base32hex(input: &str, expected: bool) {
        assert_eq!(is_base32hex(input), expected);
    }

    #[test_case(TEST_BASE32HEX, Some(TEST_STRING) ; "decode valid base32hex")]
    #[test_case("XYZ!!!", None ; "decode invalid base32hex")]
    fn test_from_base32hex(input: &str, expected: Option<&str>) {
        assert_eq!(
            from_base32hex(input).unwrap(),
            expected.map(std::string::ToString::to_string)
        );
    }

    #[test_case(TEST_BASE16_UPPER, true ; "valid base16 uppercase")]
    #[test_case(TEST_BASE16_LOWER, true ; "valid base16 lowercase")]
    #[test_case(TEST_BASE16_MIXED, true ; "valid base16 mixed case")]
    #[test_case("not valid", false ; "invalid base16 text")]
    #[test_case("GHIJ", false ; "base16 invalid chars G-J")]
    fn test_is_base16(input: &str, expected: bool) {
        assert_eq!(is_base16(input), expected);
    }

    #[test_case(TEST_BASE16_UPPER, Some(TEST_STRING) ; "decode base16 uppercase")]
    #[test_case(TEST_BASE16_LOWER, Some(TEST_STRING) ; "decode base16 lowercase")]
    #[test_case(TEST_BASE16_MIXED, Some(TEST_STRING) ; "decode base16 mixed")]
    #[test_case("GHIJ", None ; "decode invalid base16")]
    fn test_from_base16(input: &str, expected: Option<&str>) {
        assert_eq!(
            from_base16(input).unwrap(),
            expected.map(std::string::ToString::to_string)
        );
    }
}
