//! Compare the shared email source worker with the exact unmodified archive.
//! Compile with `--extern email_address=... --extern email_address_reference=...`.
use email_address::{EmailAddress, Options, ParseStorageError};
use email_address_reference as reference;

fn main() {
    let locals = [
        "name",
        "",
        "\"\"",
        "\"quoted @ local\"",
        "\"quote\\\"inside\"",
        ".name",
        "name.",
        "a..b",
        "a+b",
        "a_b",
        "δοκιμή",
        "\"δοκιμή\"",
        "name name",
        "a(b)",
        "a\nb",
        "a\0b",
    ];
    let domains = [
        "example.org",
        "localhost",
        "",
        ".org",
        "example.",
        "a..org",
        "-name.org",
        "name-.org",
        "xn--bcher-kva.example",
        "bücher.example",
        "[127.0.0.1]",
        "[IPv6:::1]",
        "[2001:db8::1]",
        "[invalid]",
        "a_b.org",
        "例子.测试",
    ];
    let mut compared = 0;
    for minimum_sub_domains in 0..=2 {
        for allow_domain_literal in [false, true] {
            for allow_display_text in [false, true] {
                for local in locals {
                    for domain in domains {
                        let email = format!("{local}@{domain}");
                        for input in [email.clone(), format!("Display Name <{email}>")] {
                            let options = Options {
                                minimum_sub_domains,
                                allow_domain_literal,
                                allow_display_text,
                            };
                            let expected = reference::EmailAddress::parse_with_options(
                                &input,
                                reference::Options {
                                    minimum_sub_domains,
                                    allow_domain_literal,
                                    allow_display_text,
                                },
                            );
                            let actual = EmailAddress::parse_with_options_and_reservation(
                                &input,
                                options,
                                |_| Ok::<_, std::convert::Infallible>(()),
                            );
                            let actual = match actual {
                                Ok(value) => Ok(value),
                                Err(ParseStorageError::Syntax(error)) => Err(error),
                                Err(error) => panic!("unexpected storage failure: {error:?}"),
                            };
                            assert_eq!(
                                format!("{:?}", expected.map(|value| value.to_string())),
                                format!("{:?}", actual.map(|value| value.to_string())),
                                "{input:?}, {options:?}"
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    println!("{compared} exact email parse comparisons passed");
}
