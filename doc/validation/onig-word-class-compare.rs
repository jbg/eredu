//! Standalone census with onig_probe={package="onig",version="=6.5.3",default-features=false}
//! and fancy_probe={package="fancy-regex",version="=0.19.0"}.
//! Record Cargo's resolved versions and the Unicode table versions with results.
fn main() {
    let onig_positive = onig_probe::Regex::new(r"\w").unwrap();
    let onig_negative = onig_probe::Regex::new(r"[^\w\s]").unwrap();
    let fancy_positive = fancy_probe::Regex::new(r"[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}]").unwrap();
    let fancy_negative = fancy_probe::Regex::new(r"[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\s]").unwrap();
    let mut positive_extra = Vec::new();
    let mut positive_missing = Vec::new();
    let mut negative_extra = Vec::new();
    let mut negative_missing = Vec::new();
    let mut scalars = 0;
    for c in (0..=0x10ffff).filter_map(char::from_u32) {
        let mut b = [0; 4];
        let text = c.encode_utf8(&mut b);
        scalars += 1;
        match (
            onig_positive.is_match(text),
            fancy_positive.is_match(text).unwrap(),
        ) {
            (true, false) => positive_extra.push(c as u32),
            (false, true) => positive_missing.push(c as u32),
            _ => {}
        }
        match (
            onig_negative.is_match(text),
            fancy_negative.is_match(text).unwrap(),
        ) {
            (true, false) => negative_extra.push(c as u32),
            (false, true) => negative_missing.push(c as u32),
            _ => {}
        }
    }
    println!("scalars={scalars} positive_extra={positive_extra:X?} positive_missing={positive_missing:X?} negative_extra={negative_extra:X?} negative_missing={negative_missing:X?}");
    assert_eq!(positive_extra, [0xb2, 0xb3, 0xb9, 0xbc, 0xbd, 0xbe]);
    assert!(
        positive_missing.is_empty() && negative_extra.is_empty() && negative_missing.is_empty()
    );
}
