fn main() {
    let mut cases=0;
    let fixed=["0","-0","0.0","-0.0","1e5000","1e-5000","0.84551240822557006","2.2250738585072014e-308","1.7976931348623157e308","1.00000000000000000000000000000000000000000000000000000000000000001","1.","1e+","--1","00"];
    for text in fixed.into_iter().map(str::to_owned).chain((-340..=340).flat_map(|e|[format!("1.234567890123456789012345678901234567890123456789e{e}"),format!("-12345678901234567890e{e}")])) {
        let plan=local::bounded_number::F64Plan::prepare(&text,0..text.len()).unwrap();
        assert!(plan.requirements().required_bytes() > 0);
        let expected=reference::from_str::<f64>(&text);
        let actual=plan.parse();
        match (expected,actual) { (Ok(a),Ok(b))=>assert_eq!(a.to_bits(),b.to_bits(),"{text}"),
            (Err(a),Err(local::bounded_number::F64Error::Source(b)))=>assert_eq!(a.to_string(),b.to_string(),"{text}"),
            (a,b)=>panic!("different typed parsing {text}: {a:?} {b:?}") }
        cases+=1;
    }
    println!("typed f64 pristine bits and diagnostics: {cases} cases");
}
