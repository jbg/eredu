// Independent pinned upstream library is linked only into this validation executable.
fn main() {
    let bases = [
        "https://example.com/a/b?query",
        "urn:opaque",
        "foo:/",
        "https://例.example/é/",
        "a://[::ffff:5:9]/",
    ];
    let schemes = ["", "HTTP://", "https://", "foo:", "//"];
    let hosts = [
        "",
        "EXAMPLE.COM",
        "[::1]",
        "[::ffff:5:9]",
        "例.example",
        "user:pw@host:80",
    ];
    let paths = [
        "",
        "/",
        "/a/../b",
        "/./%2e/..%2e/x",
        "/é/%C3%A9/水",
        "/%FF/%41/%3a",
        "/.//@@",
        "relative/../path",
    ];
    let suffixes = ["", "?a=%2f#fragment", "#é", "?q=水", "#", "?", "%"];
    let mut checked = 0usize;
    for scheme in schemes {
        for host in hosts {
            for path in paths {
                for suffix in suffixes {
                    let input = format!("{scheme}{host}{path}{suffix}");
                    let expected = fluent_uri_reference::IriRef::parse(input.as_str())
                        .map(|value| value.normalize().into_string())
                        .map_err(|error| error.to_string());
                    let actual = fluent_uri::IriRef::parse(input.as_str())
                        .map(|value| value.normalize().into_string())
                        .map_err(|error| error.to_string());
                    assert_eq!(actual, expected, "normalization: {input}");
                    let expected_ascii = fluent_uri_reference::UriRef::parse(input.as_str())
                        .map(|value| value.normalize().into_string())
                        .map_err(|error| error.to_string());
                    let actual_ascii = fluent_uri::UriRef::parse(input.as_str())
                        .map(|value| value.normalize().into_string())
                        .map_err(|error| error.to_string());
                    assert_eq!(actual_ascii, expected_ascii, "ASCII normalization: {input}");
                    for base in bases {
                        let expected = fluent_uri_reference::IriRef::parse(input.as_str())
                            .map_err(|error| error.to_string())
                            .and_then(|value| {
                                value
                                    .resolve_against(
                                        &fluent_uri_reference::Iri::parse(base).unwrap(),
                                    )
                                    .map(|value| value.into_string())
                                    .map_err(|error| error.to_string())
                            });
                        let actual = fluent_uri::IriRef::parse(input.as_str())
                            .map_err(|error| error.to_string())
                            .and_then(|value| {
                                value
                                    .resolve_against(&fluent_uri::Iri::parse(base).unwrap())
                                    .map(|value| value.into_string())
                                    .map_err(|error| error.to_string())
                            });
                        assert_eq!(actual, expected, "resolution: {base} + {input}");
                        checked += 1;
                    }
                    checked += 2;
                }
            }
        }
    }
    println!(
        "{checked} exact URI/IRI normalization, resolution and diagnostic comparisons passed."
    );
}
