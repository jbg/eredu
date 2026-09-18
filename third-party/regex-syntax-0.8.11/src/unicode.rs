use alloc::string::String;

#[cfg(all(test, feature = "unicode-case"))]
use alloc::vec::Vec;

use crate::{
    allocation::{AllocationError, Allocator},
    hir,
};

/// An inclusive range of codepoints from a generated file (hence the static
/// lifetime).
type Range = &'static [(char, char)];

/// An error that occurs when dealing with Unicode.
///
/// We don't impl the Error trait here because these always get converted
/// into other public errors. (This error type isn't exported.)
#[derive(Debug)]
pub enum Error {
    Allocation(AllocationError),
    PropertyNotFound,
    PropertyValueNotFound,
    // Not used when unicode-perl is enabled.
    #[allow(dead_code)]
    PerlClassNotFound,
}

/// An error that occurs when Unicode-aware simple case folding fails.
///
/// This error can occur when the case mapping tables necessary for Unicode
/// aware case folding are unavailable. This only occurs when the
/// `unicode-case` feature is disabled. (The feature is enabled by default.)
/// A caller-funded operation also returns the fixed allocation refusal when
/// the range storage cannot be admitted.
#[derive(Debug)]
pub enum CaseFoldError {
    /// Unicode case tables were not compiled in.
    Unavailable,
    /// A reached range allocation was refused.
    Allocation(AllocationError),
}

impl From<AllocationError> for CaseFoldError {
    fn from(error: AllocationError) -> Self {
        Self::Allocation(error)
    }
}

impl From<AllocationError> for Error {
    fn from(error: AllocationError) -> Self {
        Self::Allocation(error)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CaseFoldError {}

impl core::fmt::Display for CaseFoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("Unicode-aware case folding is not available (probably because the unicode-case feature is not enabled)"),
            Self::Allocation(error) => error.fmt(f),
        }
    }
}

/// An error that occurs when the Unicode-aware `\w` class is unavailable.
///
/// This error can occur when the data tables necessary for the Unicode aware
/// Perl character class `\w` are unavailable. This only occurs when the
/// `unicode-perl` feature is disabled. (The feature is enabled by default.)
#[derive(Debug)]
pub struct UnicodeWordError(());

#[cfg(feature = "std")]
impl std::error::Error for UnicodeWordError {}

impl core::fmt::Display for UnicodeWordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Unicode-aware \\w class is not available \
             (probably because the unicode-perl feature is not enabled)"
        )
    }
}

/// A state oriented traverser of the simple case folding table.
///
/// A case folder can be constructed via `SimpleCaseFolder::new()`, which will
/// return an error if the underlying case folding table is unavailable.
///
/// After construction, it is expected that callers will use
/// `SimpleCaseFolder::mapping` by calling it with codepoints in strictly
/// increasing order. For example, calling it on `b` and then on `a` is illegal
/// and will result in a panic.
///
/// The main idea of this type is that it tries hard to make mapping lookups
/// fast by exploiting the structure of the underlying table, and the ordering
/// assumption enables this.
#[derive(Debug)]
pub struct SimpleCaseFolder {
    /// The simple case fold table. It's a sorted association list, where the
    /// keys are Unicode scalar values and the values are the corresponding
    /// equivalence class (not including the key) of the "simple" case folded
    /// Unicode scalar values.
    table: &'static [(char, &'static [char])],
    /// The last codepoint that was used for a lookup.
    last: Option<char>,
    /// The index to the entry in `table` corresponding to the smallest key `k`
    /// such that `k > k0`, where `k0` is the most recent key lookup. Note that
    /// in particular, `k0` may not be in the table!
    next: usize,
}

impl SimpleCaseFolder {
    /// Create a new simple case folder, returning an error if the underlying
    /// case folding table is unavailable.
    pub fn new() -> Result<SimpleCaseFolder, CaseFoldError> {
        #[cfg(not(feature = "unicode-case"))]
        {
            Err(CaseFoldError::Unavailable)
        }
        #[cfg(feature = "unicode-case")]
        {
            Ok(SimpleCaseFolder {
                table: crate::unicode_tables::case_folding_simple::CASE_FOLDING_SIMPLE,
                last: None,
                next: 0,
            })
        }
    }

    /// Return the equivalence class of case folded codepoints for the given
    /// codepoint. The equivalence class returned never includes the codepoint
    /// given. If the given codepoint has no case folded codepoints (i.e.,
    /// no entry in the underlying case folding table), then this returns an
    /// empty slice.
    ///
    /// # Panics
    ///
    /// This panics when called with a `c` that is less than or equal to the
    /// previous call. In other words, callers need to use this method with
    /// strictly increasing values of `c`.
    pub fn mapping(&mut self, c: char) -> &'static [char] {
        if let Some(last) = self.last {
            assert!(
                last < c,
                "got codepoint U+{:X} which occurs before \
                 last codepoint U+{:X}",
                u32::from(c),
                u32::from(last),
            );
        }
        self.last = Some(c);
        if self.next >= self.table.len() {
            return &[];
        }
        let (k, v) = self.table[self.next];
        if k == c {
            self.next += 1;
            return v;
        }
        match self.get(c) {
            Err(i) => {
                self.next = i;
                &[]
            }
            Ok(i) => {
                // Since we require lookups to proceed
                // in order, anything we find should be
                // after whatever we thought might be
                // next. Otherwise, the caller is either
                // going out of order or we would have
                // found our next key at 'self.next'.
                assert!(i > self.next);
                self.next = i + 1;
                self.table[i].1
            }
        }
    }

    /// Returns true if and only if the given range overlaps with any region
    /// of the underlying case folding table. That is, when true, there exists
    /// at least one codepoint in the inclusive range `[start, end]` that has
    /// a non-trivial equivalence class of case folded codepoints. Conversely,
    /// when this returns false, all codepoints in the range `[start, end]`
    /// correspond to the trivial equivalence class of case folded codepoints,
    /// i.e., itself.
    ///
    /// This is useful to call before iterating over the codepoints in the
    /// range and looking up the mapping for each. If you know none of the
    /// mappings will return anything, then you might be able to skip doing it
    /// altogether.
    ///
    /// # Panics
    ///
    /// This panics when `end < start`.
    pub fn overlaps(&self, start: char, end: char) -> bool {
        use core::cmp::Ordering;

        assert!(start <= end);
        self.table
            .binary_search_by(|&(c, _)| {
                if start <= c && c <= end {
                    Ordering::Equal
                } else if c > end {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
            .is_ok()
    }

    /// Returns the index at which `c` occurs in the simple case fold table. If
    /// `c` does not occur, then this returns an `i` such that `table[i-1].0 <
    /// c` and `table[i].0 > c`.
    fn get(&self, c: char) -> Result<usize, usize> {
        self.table.binary_search_by_key(&c, |&(c1, _)| c1)
    }
}

/// A query for finding a character class defined by Unicode. This supports
/// either use of a property name directly, or lookup by property value. The
/// former generally refers to Binary properties (see UTS#44, Table 8), but
/// as a special exception (see UTS#18, Section 1.2) both general categories
/// (an enumeration) and scripts (a catalog) are supported as if each of their
/// possible values were a binary property.
///
/// In all circumstances, property names and values are normalized and
/// canonicalized. That is, `GC == gc == GeneralCategory == general_category`.
///
/// The lifetime `'a` refers to the shorter of the lifetimes of property name
/// and property value.
#[derive(Debug, Clone, Copy)]
pub enum ClassQuery<'a> {
    /// Return a class corresponding to a Unicode binary property, named by
    /// a single letter.
    OneLetter(char),
    /// Return a class corresponding to a Unicode binary property.
    ///
    /// Note that, by special exception (see UTS#18, Section 1.2), both
    /// general category values and script values are permitted here as if
    /// they were a binary property.
    Binary(&'a str),
    /// Return a class corresponding to all codepoints whose property
    /// (identified by `property_name`) corresponds to the given value
    /// (identified by `property_value`).
    ByValue {
        /// A property name.
        property_name: &'a str,
        /// A property value.
        property_value: &'a str,
    },
}

impl<'a> ClassQuery<'a> {
    #[cfg(all(test, feature = "unicode-gencat"))]
    fn canonicalize(&self) -> Result<CanonicalClassQuery, Error> {
        self.canonicalize_with_allocations(Allocator::unenforced())
    }

    fn canonicalize_with_allocations(
        &self,
        allocations: Allocator<'_>,
    ) -> Result<CanonicalClassQuery, Error> {
        match *self {
            ClassQuery::OneLetter(c) => {
                self.canonical_binary(c.encode_utf8(&mut [0; 4]), allocations)
            }
            ClassQuery::Binary(name) => self.canonical_binary(name, allocations),
            ClassQuery::ByValue {
                property_name,
                property_value,
            } => {
                let property_name =
                    symbolic_name_normalize_with_allocations(property_name, allocations)?;
                let property_value =
                    symbolic_name_normalize_with_allocations(property_value, allocations)?;

                let canon_name = match canonical_prop(&property_name)? {
                    None => return Err(Error::PropertyNotFound),
                    Some(canon_name) => canon_name,
                };
                Ok(match canon_name {
                    "General_Category" => {
                        let canon = match canonical_gencat(&property_value)? {
                            None => return Err(Error::PropertyValueNotFound),
                            Some(canon) => canon,
                        };
                        CanonicalClassQuery::GeneralCategory(canon)
                    }
                    "Script" => {
                        let canon = match canonical_script(&property_value)? {
                            None => return Err(Error::PropertyValueNotFound),
                            Some(canon) => canon,
                        };
                        CanonicalClassQuery::Script(canon)
                    }
                    _ => {
                        let vals = match property_values(canon_name)? {
                            None => return Err(Error::PropertyValueNotFound),
                            Some(vals) => vals,
                        };
                        let canon_val = match canonical_value(vals, &property_value) {
                            None => return Err(Error::PropertyValueNotFound),
                            Some(canon_val) => canon_val,
                        };
                        CanonicalClassQuery::ByValue {
                            property_name: canon_name,
                            property_value: canon_val,
                        }
                    }
                })
            }
        }
    }

    fn canonical_binary(
        &self,
        name: &str,
        allocations: Allocator<'_>,
    ) -> Result<CanonicalClassQuery, Error> {
        let norm = symbolic_name_normalize_with_allocations(name, allocations)?;

        // This is a special case where 'cf' refers to the 'Format' general
        // category, but where the 'cf' abbreviation is also an abbreviation
        // for the 'Case_Folding' property. But we want to treat it as
        // a general category. (Currently, we don't even support the
        // 'Case_Folding' property. But if we do in the future, users will be
        // required to spell it out.)
        //
        // Also 'sc' refers to the 'Currency_Symbol' general category, but is
        // also the abbreviation for the 'Script' property. So we avoid calling
        // 'canonical_prop' for it too, which would erroneously normalize it
        // to 'Script'.
        //
        // Another case: 'lc' is an abbreviation for the 'Cased_Letter'
        // general category, but is also an abbreviation for the 'Lowercase_Mapping'
        // property. We don't currently support the latter, so as with 'cf'
        // above, we treat 'lc' as 'Cased_Letter'.
        if norm != "cf" && norm != "sc" && norm != "lc" {
            if let Some(canon) = canonical_prop(&norm)? {
                return Ok(CanonicalClassQuery::Binary(canon));
            }
        }
        if let Some(canon) = canonical_gencat(&norm)? {
            return Ok(CanonicalClassQuery::GeneralCategory(canon));
        }
        if let Some(canon) = canonical_script(&norm)? {
            return Ok(CanonicalClassQuery::Script(canon));
        }
        Err(Error::PropertyNotFound)
    }
}

/// Like ClassQuery, but its parameters have been canonicalized. This also
/// differentiates binary properties from flattened general categories and
/// scripts.
#[derive(Debug, Eq, PartialEq)]
enum CanonicalClassQuery {
    /// The canonical binary property name.
    Binary(&'static str),
    /// The canonical general category name.
    GeneralCategory(&'static str),
    /// The canonical script name.
    Script(&'static str),
    /// An arbitrary association between property and value, both of which
    /// have been canonicalized.
    ///
    /// Note that by construction, the property name of ByValue will never
    /// be General_Category or Script. Those two cases are subsumed by the
    /// eponymous variants.
    ByValue {
        /// The canonical property name.
        property_name: &'static str,
        /// The canonical property value.
        property_value: &'static str,
    },
}

/// Looks up a Unicode class given a query. If one doesn't exist, then
/// `None` is returned.
#[cfg(test)]
pub fn class(query: ClassQuery<'_>) -> Result<hir::ClassUnicode, Error> {
    class_with_allocations(query, Allocator::unenforced())
}

pub fn class_with_allocations(
    query: ClassQuery<'_>,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    use self::CanonicalClassQuery::*;

    match query.canonicalize_with_allocations(allocations)? {
        Binary(name) => bool_property(name, allocations),
        GeneralCategory(name) => gencat(name, allocations),
        Script(name) => script(name, allocations),
        ByValue {
            property_name: "Age",
            property_value,
        } => {
            let mut class = hir::ClassUnicode::empty();
            for set in ages(property_value)? {
                class.union_with_allocations(
                    &hir_class_with_allocations(set, allocations)?,
                    allocations,
                )?;
            }
            Ok(class)
        }
        ByValue {
            property_name: "Script_Extensions",
            property_value,
        } => script_extension(property_value, allocations),
        ByValue {
            property_name: "Grapheme_Cluster_Break",
            property_value,
        } => gcb(property_value, allocations),
        ByValue {
            property_name: "Sentence_Break",
            property_value,
        } => sb(property_value, allocations),
        ByValue {
            property_name: "Word_Break",
            property_value,
        } => wb(property_value, allocations),
        _ => {
            // What else should we support?
            Err(Error::PropertyNotFound)
        }
    }
}

/// Returns a Unicode aware class for \w.
///
/// This returns an error if the data is not available for \w.
#[cfg(test)]
pub fn perl_word() -> Result<hir::ClassUnicode, Error> {
    perl_word_with_allocations(Allocator::unenforced())
}

pub fn perl_word_with_allocations(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-perl"))]
    fn imp(_allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PerlClassNotFound)
    }

    #[cfg(feature = "unicode-perl")]
    fn imp(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::perl_word::PERL_WORD;
        Ok(hir_class_with_allocations(PERL_WORD, allocations)?)
    }

    imp(allocations)
}

/// Returns a Unicode aware class for \s.
///
/// This returns an error if the data is not available for \s.

pub fn perl_space_with_allocations(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(any(feature = "unicode-perl", feature = "unicode-bool")))]
    fn imp(_allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PerlClassNotFound)
    }

    #[cfg(all(feature = "unicode-perl", not(feature = "unicode-bool")))]
    fn imp(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::perl_space::WHITE_SPACE;
        Ok(hir_class_with_allocations(WHITE_SPACE, allocations)?)
    }

    #[cfg(feature = "unicode-bool")]
    fn imp(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::property_bool::WHITE_SPACE;
        Ok(hir_class_with_allocations(WHITE_SPACE, allocations)?)
    }

    imp(allocations)
}

/// Returns a Unicode aware class for \d.
///
/// This returns an error if the data is not available for \d.

pub fn perl_digit_with_allocations(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(any(feature = "unicode-perl", feature = "unicode-gencat")))]
    fn imp(_allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PerlClassNotFound)
    }

    #[cfg(all(feature = "unicode-perl", not(feature = "unicode-gencat")))]
    fn imp(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::perl_decimal::DECIMAL_NUMBER;
        Ok(hir_class_with_allocations(DECIMAL_NUMBER, allocations)?)
    }

    #[cfg(feature = "unicode-gencat")]
    fn imp(allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::general_category::DECIMAL_NUMBER;
        Ok(hir_class_with_allocations(DECIMAL_NUMBER, allocations)?)
    }

    imp(allocations)
}

/// Build a Unicode HIR class from a sequence of Unicode scalar value ranges.

pub fn hir_class_with_allocations(
    ranges: &[(char, char)],
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, AllocationError> {
    hir::ClassUnicode::new_with_allocations(
        ranges
            .iter()
            .map(|&(s, e)| hir::ClassUnicodeRange::new(s, e)),
        allocations,
    )
}

/// Returns true only if the given codepoint is in the `\w` character class.
///
/// If the `unicode-perl` feature is not enabled, then this returns an error.
pub fn is_word_character(c: char) -> Result<bool, UnicodeWordError> {
    #[cfg(not(feature = "unicode-perl"))]
    fn imp(_: char) -> Result<bool, UnicodeWordError> {
        Err(UnicodeWordError(()))
    }

    #[cfg(feature = "unicode-perl")]
    fn imp(c: char) -> Result<bool, UnicodeWordError> {
        use crate::{is_word_byte, unicode_tables::perl_word::PERL_WORD};

        if u8::try_from(c).map_or(false, is_word_byte) {
            return Ok(true);
        }
        Ok(PERL_WORD
            .binary_search_by(|&(start, end)| {
                use core::cmp::Ordering;

                if start <= c && c <= end {
                    Ordering::Equal
                } else if start > c {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
            .is_ok())
    }

    imp(c)
}

/// A mapping of property values for a specific property.
///
/// The first element of each tuple is a normalized property value while the
/// second element of each tuple is the corresponding canonical property
/// value.
type PropertyValues = &'static [(&'static str, &'static str)];

fn canonical_gencat(normalized_value: &str) -> Result<Option<&'static str>, Error> {
    Ok(match normalized_value {
        "any" => Some("Any"),
        "assigned" => Some("Assigned"),
        "ascii" => Some("ASCII"),
        _ => {
            let gencats = property_values("General_Category")?.unwrap();
            canonical_value(gencats, normalized_value)
        }
    })
}

fn canonical_script(normalized_value: &str) -> Result<Option<&'static str>, Error> {
    let scripts = property_values("Script")?.unwrap();
    Ok(canonical_value(scripts, normalized_value))
}

/// Find the canonical property name for the given normalized property name.
///
/// If no such property exists, then `None` is returned.
///
/// The normalized property name must have been normalized according to
/// UAX44 LM3, which can be done using `symbolic_name_normalize`.
///
/// If the property names data is not available, then an error is returned.
fn canonical_prop(normalized_name: &str) -> Result<Option<&'static str>, Error> {
    #[cfg(not(any(
        feature = "unicode-age",
        feature = "unicode-bool",
        feature = "unicode-gencat",
        feature = "unicode-perl",
        feature = "unicode-script",
        feature = "unicode-segment",
    )))]
    fn imp(_: &str) -> Result<Option<&'static str>, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(any(
        feature = "unicode-age",
        feature = "unicode-bool",
        feature = "unicode-gencat",
        feature = "unicode-perl",
        feature = "unicode-script",
        feature = "unicode-segment",
    ))]
    fn imp(name: &str) -> Result<Option<&'static str>, Error> {
        use crate::unicode_tables::property_names::PROPERTY_NAMES;

        Ok(PROPERTY_NAMES
            .binary_search_by_key(&name, |&(n, _)| n)
            .ok()
            .map(|i| PROPERTY_NAMES[i].1))
    }

    imp(normalized_name)
}

/// Find the canonical property value for the given normalized property
/// value.
///
/// The given property values should correspond to the values for the property
/// under question, which can be found using `property_values`.
///
/// If no such property value exists, then `None` is returned.
///
/// The normalized property value must have been normalized according to
/// UAX44 LM3, which can be done using `symbolic_name_normalize`.
fn canonical_value(vals: PropertyValues, normalized_value: &str) -> Option<&'static str> {
    vals.binary_search_by_key(&normalized_value, |&(n, _)| n)
        .ok()
        .map(|i| vals[i].1)
}

/// Return the table of property values for the given property name.
///
/// If the property values data is not available, then an error is returned.
fn property_values(canonical_property_name: &'static str) -> Result<Option<PropertyValues>, Error> {
    #[cfg(not(any(
        feature = "unicode-age",
        feature = "unicode-bool",
        feature = "unicode-gencat",
        feature = "unicode-perl",
        feature = "unicode-script",
        feature = "unicode-segment",
    )))]
    fn imp(_: &'static str) -> Result<Option<PropertyValues>, Error> {
        Err(Error::PropertyValueNotFound)
    }

    #[cfg(any(
        feature = "unicode-age",
        feature = "unicode-bool",
        feature = "unicode-gencat",
        feature = "unicode-perl",
        feature = "unicode-script",
        feature = "unicode-segment",
    ))]
    fn imp(name: &'static str) -> Result<Option<PropertyValues>, Error> {
        use crate::unicode_tables::property_values::PROPERTY_VALUES;

        Ok(PROPERTY_VALUES
            .binary_search_by_key(&name, |&(n, _)| n)
            .ok()
            .map(|i| PROPERTY_VALUES[i].1))
    }

    imp(canonical_property_name)
}

// This is only used in some cases, but small enough to just let it be dead
// instead of figuring out (and maintaining) the right set of features.
#[allow(dead_code)]
fn property_set(
    name_map: &'static [(&'static str, Range)],
    canonical: &'static str,
) -> Option<Range> {
    name_map
        .binary_search_by_key(&canonical, |x| x.0)
        .ok()
        .map(|i| name_map[i].1)
}

/// Returns an iterator over Unicode Age sets. Each item corresponds to a set
/// of codepoints that were added in a particular revision of Unicode. The
/// iterator yields items in chronological order.
///
/// If the given age value isn't valid or if the data isn't available, then an
/// error is returned instead.
fn ages(canonical_age: &str) -> Result<impl Iterator<Item = Range>, Error> {
    #[cfg(not(feature = "unicode-age"))]
    fn imp(_: &str) -> Result<impl Iterator<Item = Range>, Error> {
        use core::option::IntoIter;
        Err::<IntoIter<Range>, _>(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-age")]
    fn imp(canonical_age: &str) -> Result<impl Iterator<Item = Range>, Error> {
        use crate::unicode_tables::age;

        const AGES: &[(&str, Range)] = &[
            ("V1_1", age::V1_1),
            ("V2_0", age::V2_0),
            ("V2_1", age::V2_1),
            ("V3_0", age::V3_0),
            ("V3_1", age::V3_1),
            ("V3_2", age::V3_2),
            ("V4_0", age::V4_0),
            ("V4_1", age::V4_1),
            ("V5_0", age::V5_0),
            ("V5_1", age::V5_1),
            ("V5_2", age::V5_2),
            ("V6_0", age::V6_0),
            ("V6_1", age::V6_1),
            ("V6_2", age::V6_2),
            ("V6_3", age::V6_3),
            ("V7_0", age::V7_0),
            ("V8_0", age::V8_0),
            ("V9_0", age::V9_0),
            ("V10_0", age::V10_0),
            ("V11_0", age::V11_0),
            ("V12_0", age::V12_0),
            ("V12_1", age::V12_1),
            ("V13_0", age::V13_0),
            ("V14_0", age::V14_0),
            ("V15_0", age::V15_0),
            ("V15_1", age::V15_1),
            ("V16_0", age::V16_0),
        ];
        assert_eq!(AGES.len(), age::BY_NAME.len(), "ages are out of sync");

        let pos = AGES.iter().position(|&(age, _)| canonical_age == age);
        match pos {
            None => Err(Error::PropertyValueNotFound),
            Some(i) => Ok(AGES[..=i].iter().map(|&(_, classes)| classes)),
        }
    }

    imp(canonical_age)
}

/// Returns the Unicode HIR class corresponding to the given general category.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given general category could not be found, or if the general
/// category data is not available, then an error is returned.
fn gencat(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-gencat"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-gencat")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::general_category::BY_NAME;
        match name {
            "ASCII" => Ok(hir_class_with_allocations(&[('\0', '\x7F')], allocations)?),
            "Any" => Ok(hir_class_with_allocations(
                &[('\0', '\u{10FFFF}')],
                allocations,
            )?),
            "Assigned" => {
                let mut cls = gencat("Unassigned", allocations)?;
                cls.negate_with_allocations(allocations)?;
                Ok(cls)
            }
            name => hir_class_with_allocations(
                property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
                allocations,
            )
            .map_err(Error::from),
        }
    }

    match canonical_name {
        "Decimal_Number" => perl_digit_with_allocations(allocations),
        name => imp(name, allocations),
    }
}

/// Returns the Unicode HIR class corresponding to the given script.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given script could not be found, or if the script data is not
/// available, then an error is returned.
fn script(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-script"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-script")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::script::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    imp(canonical_name, allocations)
}

/// Returns the Unicode HIR class corresponding to the given script extension.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given script extension could not be found, or if the script data is
/// not available, then an error is returned.
fn script_extension(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-script"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-script")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::script_extension::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    imp(canonical_name, allocations)
}

/// Returns the Unicode HIR class corresponding to the given Unicode boolean
/// property.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given boolean property could not be found, or if the boolean
/// property data is not available, then an error is returned.
fn bool_property(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-bool"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-bool")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::property_bool::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    match canonical_name {
        "Decimal_Number" => perl_digit_with_allocations(allocations),
        "White_Space" => perl_space_with_allocations(allocations),
        name => imp(name, allocations),
    }
}

/// Returns the Unicode HIR class corresponding to the given grapheme cluster
/// break property.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given property could not be found, or if the corresponding data is
/// not available, then an error is returned.
fn gcb(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-segment"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-segment")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::grapheme_cluster_break::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    imp(canonical_name, allocations)
}

/// Returns the Unicode HIR class corresponding to the given word break
/// property.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given property could not be found, or if the corresponding data is
/// not available, then an error is returned.
fn wb(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-segment"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-segment")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::word_break::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    imp(canonical_name, allocations)
}

/// Returns the Unicode HIR class corresponding to the given sentence
/// break property.
///
/// Name canonicalization is assumed to be performed by the caller.
///
/// If the given property could not be found, or if the corresponding data is
/// not available, then an error is returned.
fn sb(
    canonical_name: &'static str,
    allocations: Allocator<'_>,
) -> Result<hir::ClassUnicode, Error> {
    #[cfg(not(feature = "unicode-segment"))]
    fn imp(_: &'static str, _allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        Err(Error::PropertyNotFound)
    }

    #[cfg(feature = "unicode-segment")]
    fn imp(name: &'static str, allocations: Allocator<'_>) -> Result<hir::ClassUnicode, Error> {
        use crate::unicode_tables::sentence_break::BY_NAME;
        hir_class_with_allocations(
            property_set(BY_NAME, name).ok_or(Error::PropertyValueNotFound)?,
            allocations,
        )
        .map_err(Error::from)
    }

    imp(canonical_name, allocations)
}

/// Like symbolic_name_normalize_bytes, but operates on a string.
#[cfg(test)]
fn symbolic_name_normalize(x: &str) -> String {
    symbolic_name_normalize_with_allocations(x, Allocator::unenforced())
        .expect("ordinary normalization allocation")
}

fn symbolic_name_normalize_with_allocations(
    x: &str,
    allocations: Allocator<'_>,
) -> Result<String, AllocationError> {
    let mut tmp = allocations.copy_slice(x.as_bytes())?;
    let len = symbolic_name_normalize_bytes(&mut tmp).len();
    tmp.truncate(len);
    // This should always succeed because `symbolic_name_normalize_bytes`
    // guarantees that `&tmp[..len]` is always valid UTF-8.
    //
    // N.B. We could avoid the additional UTF-8 check here, but it's unlikely
    // to be worth skipping the additional safety check. A benchmark must
    // justify it first.
    Ok(String::from_utf8(tmp).unwrap())
}

/// Normalize the given symbolic name in place according to UAX44-LM3.
///
/// A "symbolic name" typically corresponds to property names and property
/// value aliases. Note, though, that it should not be applied to property
/// string values.
///
/// The slice returned is guaranteed to be valid UTF-8 for all possible values
/// of `slice`.
///
/// See: https://unicode.org/reports/tr44/#UAX44-LM3
fn symbolic_name_normalize_bytes(slice: &mut [u8]) -> &mut [u8] {
    // I couldn't find a place in the standard that specified that property
    // names/aliases had a particular structure (unlike character names), but
    // we assume that it's ASCII only and drop anything that isn't ASCII.
    let mut start = 0;
    let mut starts_with_is = false;
    if slice.len() >= 2 {
        // Ignore any "is" prefix.
        starts_with_is = slice[0..2] == b"is"[..]
            || slice[0..2] == b"IS"[..]
            || slice[0..2] == b"iS"[..]
            || slice[0..2] == b"Is"[..];
        if starts_with_is {
            start = 2;
        }
    }
    let mut next_write = 0;
    for i in start..slice.len() {
        // VALIDITY ARGUMENT: To guarantee that the resulting slice is valid
        // UTF-8, we ensure that the slice contains only ASCII bytes. In
        // particular, we drop every non-ASCII byte from the normalized string.
        let b = slice[i];
        if b == b' ' || b == b'_' || b == b'-' {
            continue;
        } else if b'A' <= b && b <= b'Z' {
            slice[next_write] = b + (b'a' - b'A');
            next_write += 1;
        } else if b <= 0x7F {
            slice[next_write] = b;
            next_write += 1;
        }
    }
    // Special case: ISO_Comment has a 'isc' abbreviation. Since we generally
    // ignore 'is' prefixes, the 'isc' abbreviation gets caught in the cross
    // fire and ends up creating an alias for 'c' to 'ISO_Comment', but it
    // is actually an alias for the 'Other' general category.
    if starts_with_is && next_write == 1 && slice[0] == b'c' {
        slice[0] = b'i';
        slice[1] = b's';
        slice[2] = b'c';
        next_write = 3;
    }
    &mut slice[..next_write]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "unicode-case")]
    fn simple_fold_ok(c: char) -> impl Iterator<Item = char> {
        SimpleCaseFolder::new().unwrap().mapping(c).iter().copied()
    }

    #[cfg(feature = "unicode-case")]
    fn contains_case_map(start: char, end: char) -> bool {
        SimpleCaseFolder::new().unwrap().overlaps(start, end)
    }

    #[test]
    #[cfg(feature = "unicode-case")]
    fn simple_fold_k() {
        let xs: Vec<char> = simple_fold_ok('k').collect();
        assert_eq!(xs, alloc::vec!['K', 'K']);

        let xs: Vec<char> = simple_fold_ok('K').collect();
        assert_eq!(xs, alloc::vec!['k', 'K']);

        let xs: Vec<char> = simple_fold_ok('K').collect();
        assert_eq!(xs, alloc::vec!['K', 'k']);
    }

    #[test]
    #[cfg(feature = "unicode-case")]
    fn simple_fold_a() {
        let xs: Vec<char> = simple_fold_ok('a').collect();
        assert_eq!(xs, alloc::vec!['A']);

        let xs: Vec<char> = simple_fold_ok('A').collect();
        assert_eq!(xs, alloc::vec!['a']);
    }

    #[test]
    #[cfg(not(feature = "unicode-case"))]
    fn simple_fold_disabled() {
        assert!(SimpleCaseFolder::new().is_err());
    }

    #[test]
    #[cfg(feature = "unicode-case")]
    fn range_contains() {
        assert!(contains_case_map('A', 'A'));
        assert!(contains_case_map('Z', 'Z'));
        assert!(contains_case_map('A', 'Z'));
        assert!(contains_case_map('@', 'A'));
        assert!(contains_case_map('Z', '['));
        assert!(contains_case_map('☃', 'Ⰰ'));

        assert!(!contains_case_map('[', '['));
        assert!(!contains_case_map('[', '`'));

        assert!(!contains_case_map('☃', '☃'));
    }

    #[test]
    #[cfg(feature = "unicode-gencat")]
    fn regression_466() {
        use super::{CanonicalClassQuery, ClassQuery};

        let q = ClassQuery::OneLetter('C');
        assert_eq!(
            q.canonicalize().unwrap(),
            CanonicalClassQuery::GeneralCategory("Other")
        );
    }

    #[test]
    fn sym_normalize() {
        let sym_norm = symbolic_name_normalize;

        assert_eq!(sym_norm("Line_Break"), "linebreak");
        assert_eq!(sym_norm("Line-break"), "linebreak");
        assert_eq!(sym_norm("linebreak"), "linebreak");
        assert_eq!(sym_norm("BA"), "ba");
        assert_eq!(sym_norm("ba"), "ba");
        assert_eq!(sym_norm("Greek"), "greek");
        assert_eq!(sym_norm("isGreek"), "greek");
        assert_eq!(sym_norm("IS_Greek"), "greek");
        assert_eq!(sym_norm("isc"), "isc");
        assert_eq!(sym_norm("is c"), "isc");
        assert_eq!(sym_norm("is_c"), "isc");
    }

    #[test]
    fn valid_utf8_symbolic() {
        let mut x = b"abc\xFFxyz".to_vec();
        let y = symbolic_name_normalize_bytes(&mut x);
        assert_eq!(y, b"abcxyz");
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use crate::allocation::Allocation;
    use core::cell::Cell;
    struct RefuseAt {
        calls: Cell<usize>,
        at: usize,
    }
    impl Allocation for RefuseAt {
        fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
            assert!(bytes > 0);
            let call = self.calls.get();
            assert!(call <= self.at, "producer continued after refusal");
            self.calls.set(call + 1);
            if call == self.at {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn normalization_funds_actual_input_copy_before_work() {
        let policy = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert!(matches!(
            class_with_allocations(
                ClassQuery::Binary("General_Category"),
                Allocator::new(&policy)
            ),
            Err(Error::Allocation(AllocationError::Refused))
        ));
        assert_eq!(policy.calls.get(), 1);
    }
    #[test]
    #[cfg(all(
        feature = "unicode-age",
        feature = "unicode-bool",
        feature = "unicode-gencat",
        feature = "unicode-script",
        feature = "unicode-segment",
        feature = "unicode-perl"
    ))]
    fn property_classes_stop_at_each_refused_allocation() {
        let queries = [
            ClassQuery::OneLetter('L'),
            ClassQuery::Binary("Assigned"),
            ClassQuery::Binary("Alphabetic"),
            ClassQuery::ByValue {
                property_name: "Script",
                property_value: "Greek",
            },
            ClassQuery::ByValue {
                property_name: "Script_Extensions",
                property_value: "Greek",
            },
            ClassQuery::ByValue {
                property_name: "Age",
                property_value: "2.0",
            },
            ClassQuery::ByValue {
                property_name: "Grapheme_Cluster_Break",
                property_value: "Extend",
            },
            ClassQuery::ByValue {
                property_name: "Sentence_Break",
                property_value: "Upper",
            },
            ClassQuery::ByValue {
                property_name: "Word_Break",
                property_value: "ALetter",
            },
        ];
        for query in queries {
            let full = RefuseAt {
                calls: Cell::new(0),
                at: usize::MAX,
            };
            let actual = class_with_allocations(query, Allocator::new(&full)).unwrap();
            assert_eq!(actual, class(query).unwrap());
            assert!(!actual.ranges().is_empty());
            for at in 0..full.calls.get() {
                let quota = RefuseAt {
                    calls: Cell::new(0),
                    at,
                };
                assert!(matches!(
                    class_with_allocations(query, Allocator::new(&quota)),
                    Err(Error::Allocation(AllocationError::Refused))
                ));
                assert_eq!(quota.calls.get(), at + 1);
            }
        }
        for producer in [
            perl_word_with_allocations,
            perl_space_with_allocations,
            perl_digit_with_allocations,
        ] {
            let full = RefuseAt {
                calls: Cell::new(0),
                at: usize::MAX,
            };
            let paid = producer(Allocator::new(&full)).unwrap();
            assert_eq!(paid, producer(Allocator::unenforced()).unwrap());
            for at in 0..full.calls.get() {
                let quota = RefuseAt {
                    calls: Cell::new(0),
                    at,
                };
                assert!(matches!(
                    producer(Allocator::new(&quota)),
                    Err(Error::Allocation(AllocationError::Refused))
                ));
                assert_eq!(quota.calls.get(), at + 1);
            }
        }
    }
    #[test]
    #[cfg(feature = "unicode-case")]
    fn unicode_casefold_refusal_preserves_original_class() {
        let source = hir::ClassUnicode::new([hir::ClassUnicodeRange::new('A', 'Z')]);
        let full = RefuseAt {
            calls: Cell::new(0),
            at: usize::MAX,
        };
        let mut folded = source.clone();
        folded
            .try_case_fold_simple_with_allocations(Allocator::new(&full))
            .unwrap();
        assert!(folded
            .ranges()
            .iter()
            .any(|r| r.start() <= 'K' && 'K' <= r.end()));
        for at in 0..full.calls.get() {
            let quota = RefuseAt {
                calls: Cell::new(0),
                at,
            };
            let mut actual = source.clone();
            assert!(matches!(
                actual.try_case_fold_simple_with_allocations(Allocator::new(&quota)),
                Err(CaseFoldError::Allocation(AllocationError::Refused))
            ));
            assert_eq!(source, actual);
            assert_eq!(quota.calls.get(), at + 1);
        }
    }
    #[test]
    fn literal_conversion_and_class_clone_use_exact_producer() {
        let source = hir::ClassUnicode::new([hir::ClassUnicodeRange::new('k', 'k')]);
        let quota = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert_eq!(
            Err(AllocationError::Refused),
            source.literal_with_allocations(Allocator::new(&quota))
        );
        let quota = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert_eq!(
            Err(AllocationError::Refused),
            source.to_byte_class_with_allocations(Allocator::new(&quota))
        );
        let quota = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert_eq!(
            Err(AllocationError::Refused),
            source.clone_with_allocations(Allocator::new(&quota))
        );
        let bytes = source
            .to_byte_class_with_allocations(Allocator::unenforced())
            .unwrap()
            .unwrap();
        let quota = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert_eq!(
            Err(AllocationError::Refused),
            bytes.to_unicode_class_with_allocations(Allocator::new(&quota))
        );
        let quota = RefuseAt {
            calls: Cell::new(0),
            at: 0,
        };
        assert_eq!(
            Err(AllocationError::Refused),
            bytes.literal_with_allocations(Allocator::new(&quota))
        );
        assert_eq!(Some(b"k".to_vec()), source.literal());
        assert_eq!(Some(source), bytes.to_unicode_class());
    }
}
