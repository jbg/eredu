use email_address::{EmailAddress, Options, ParseStorageError};

#[test]
fn every_reached_reservation_precedes_its_producer() {
    for input in [
        "name@example.org",
        "\"quoted @ local\"@example.org",
        "not an address",
        "a@",
    ] {
        let expected = EmailAddress::parse_with_options(input, Options::default());
        let mut layouts = Vec::new();
        let actual =
            EmailAddress::parse_with_options_and_reservation(input, Options::default(), |layout| {
                layouts.push(layout);
                Ok::<_, usize>(())
            });
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => assert_eq!(expected, actual),
            (Err(expected), Err(ParseStorageError::Syntax(actual))) => assert_eq!(expected, actual),
            other => panic!("parser outcome changed: {other:?}"),
        }
        for refused in 0..layouts.len() {
            let mut calls = 0;
            let error =
                EmailAddress::parse_with_options_and_reservation(input, Options::default(), |_| {
                    let reached = calls;
                    calls += 1;
                    assert!(reached <= refused, "reservation after first refusal");
                    if reached == refused {
                        Err(refused)
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert!(matches!(error, ParseStorageError::Funding(actual) if actual == refused));
            assert_eq!(calls, refused + 1);
        }
    }
}
