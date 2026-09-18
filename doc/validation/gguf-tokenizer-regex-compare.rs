use fancy_regex::{workspace::construction,allocation::Unenforced};
fn main() {
    let mut texts=vec![String::new(), "I'm café 123456789 中文漢字かな ΑΣ_é\u{301}\u{200c}\u{200d}\r\n".into(), " ".repeat(4096)+"x", "世".repeat(1024)];
    let alphabet=['a','Z','é','東','🙂','\u{301}','\u{200c}','0','९',' ','\r','\n','\t','\'','/','!','\0'];
    let mut state=0x92d68ca2u32;
    for _ in 0..256 {let mut text=String::new(); for _ in 0..40 {state^=state<<13;state^=state>>17;state^=state<<5;text.push(alphabet[state as usize%alphabet.len()]);}texts.push(text);}
    let mut cases=0;
    for pattern in construction::patterns() {
        let source=construction::Plan::new(pattern).unwrap().prepare().unwrap();
        let mut workspace=source.plan().unwrap().prepare().unwrap();
        let ordinary=fancy_regex::Regex::new(pattern).unwrap();
        let reference=fancy_regex_reference::Regex::new(pattern).unwrap();
        let mut scoped=ordinary.search_workspace_with_allocations(&Unenforced).unwrap();
        for text in &texts {
            let expected:Vec<_>=reference.find_iter(text).map(|m|m.map(|m|(m.start(),m.end()))).collect();
            assert!(expected.iter().all(Result::is_ok));
            let expected:Vec<_>=expected.into_iter().map(Result::unwrap).collect();
            let actual:Vec<_>=workspace.find_iter(text).map(|m|m.map(|m|(m.start(),m.end()))).collect::<Result<_,_>>().unwrap();
            assert_eq!(actual,expected,"{pattern:?}/{text:?}");
            assert_eq!(ordinary.find_iter(text).map(|m|m.map(|m|(m.start(),m.end()))).collect::<Result<Vec<_>,_>>().unwrap(),expected);
            assert_eq!(scoped.find(text,&Unenforced).unwrap().map(|m|(m.start(),m.end())),expected.first().copied());
            cases+=3;
        }
    }
    println!("PASS {cases} exact span/first-match comparisons for9sources against pristine fancy-regex0.19.0 and untouched lower dependencies");
}
