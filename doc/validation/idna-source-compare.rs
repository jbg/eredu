//! Standalone comparison linked to the local IDNA worker and pristine pinned
//! idna/idna_adapter/icu_normalizer archive artifacts as `idna_reference`.
use std::{borrow::Cow, fmt::Write, time::Instant};
fn decode(input: &str) -> String {
    let mut output = String::new();
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            output.push(c);
            continue;
        }
        match chars.next().unwrap() {
            '\\' => output.push('\\'),
            'u' => {
                let digits: String = chars.by_ref().take(4).collect();
                let value = u32::from_str_radix(&digits, 16).unwrap();
                match char::from_u32(value) {
                    Some(c) => output.push(c),
                    None => write!(output, "\\u{digits}").unwrap(),
                }
            }
            value => panic!("unexpected fixture escape {value}"),
        }
    }
    output
}
fn shape(value: Result<Cow<'_, str>, impl std::fmt::Debug>) -> Option<(String, bool)> {
    value.ok().map(|value| {
        let borrowed = matches!(value, Cow::Borrowed(_));
        (value.into_owned(), borrowed)
    })
}
fn main() {
    use idna::uts46::{AsciiDenyList as A, DnsLength as D, Hyphens as H, Uts46 as U};
    use idna_reference::uts46::{AsciiDenyList as RA, DnsLength as RD, Hyphens as RH, Uts46 as RU};
    let path = std::env::args()
        .nth(1)
        .expect("upstream IdnaTestV2.txt path");
    let file = std::fs::read_to_string(path).unwrap();
    let mut inputs: Vec<String> = file
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| decode(line.split(';').next().unwrap().trim()))
        .collect();
    for n in [
        0, 1, 16, 17, 18, 31, 32, 33, 58, 59, 60, 127, 253, 254, 300, 1024,
    ] {
        inputs.push(format!("a{}.example", "\u{315}\u{301}\u{300}".repeat(n)));
        inputs.push(format!("{}.example", "é.".repeat(n)));
        inputs.push(format!("{}.example", "音".repeat(n)));
    }
    let mut comparisons = 0;
    for input in &inputs {
        for (deny, rdeny) in [
            (A::EMPTY, RA::EMPTY),
            (A::STD3, RA::STD3),
            (A::URL, RA::URL),
        ] {
            for (hyphens, rhyphens) in [
                (H::Allow, RH::Allow),
                (H::Check, RH::Check),
                (H::CheckFirstLast, RH::CheckFirstLast),
            ] {
                for (dns, rdns) in [
                    (D::Ignore, RD::Ignore),
                    (D::Verify, RD::Verify),
                    (D::VerifyAllowRootDot, RD::VerifyAllowRootDot),
                ] {
                    assert_eq!(
                        shape(U::new().to_ascii(input.as_bytes(), deny, hyphens, dns)),
                        shape(RU::new().to_ascii(input.as_bytes(), rdeny, rhyphens, rdns)),
                        "ASCII {input}"
                    );
                    comparisons += 1;
                }
                let (text, status) = U::new().to_unicode(input.as_bytes(), deny, hyphens);
                let (expected, prior) = RU::new().to_unicode(input.as_bytes(), rdeny, rhyphens);
                assert_eq!(
                    (&text, status.is_ok(), matches!(text, Cow::Borrowed(_))),
                    (
                        &expected,
                        prior.is_ok(),
                        matches!(expected, Cow::Borrowed(_))
                    ),
                    "Unicode {input}"
                );
                comparisons += 1;
            }
        }
    }
    println!("{comparisons} exact ASCII/Unicode value, status and borrow-shape comparisons across {} sources passed",inputs.len());
    for (name, input) in [
        ("ascii", "docs.example.com".to_owned()),
        ("unicode", "BÜCHER.例え.example".to_owned()),
        (
            "combining",
            format!("a{}.example", "\u{315}\u{301}\u{300}".repeat(100)),
        ),
        ("many_labels", format!("{}.example", "é.".repeat(140))),
    ] {
        let iterations = 2000;
        let prior = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(RU::new().to_ascii(
                std::hint::black_box(input.as_bytes()),
                RA::STD3,
                RH::Check,
                RD::Ignore,
            ))
            .ok();
        }
        let reference = prior.elapsed();
        let now = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(U::new().to_ascii(
                std::hint::black_box(input.as_bytes()),
                A::STD3,
                H::Check,
                D::Ignore,
            ))
            .ok();
        }
        let local = now.elapsed();
        println!(
            "{name} ({} bytes): local={local:?} pristine={reference:?} ratio={:.3}",
            input.len(),
            local.as_secs_f64() / reference.as_secs_f64()
        );
    }
}
