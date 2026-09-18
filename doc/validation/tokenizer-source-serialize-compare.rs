//! Standalone exact-version oracle: aliases `local` and pristine `reference`.
use std::{cell::Cell, hint::black_box, time::Instant};
use serde_json::{json, Value};
struct Meter(Cell<usize>);
impl local::source_serialization::Allocation for Meter {
    fn reserve(&self, bytes:usize)->Result<(),local::source_serialization::AllocationError>{
        self.0.set(self.0.get().checked_add(bytes).unwrap()); Ok(())
    }
}
fn document(kind:&str,n:usize)->Value {
    let mut vocab=serde_json::Map::new();
    vocab.insert("[UNK]".into(),json!(0));
    // Shuffled insertion exercises ordering independently of input object order.
    for i in (0..n).rev() {vocab.insert(format!("t{i}"),json!(i+1));}
    let model=match kind {
        "WordLevel"=>json!({"type":kind,"vocab":vocab,"unk_token":"[UNK]"}),
        "Unigram"=>{
            let mut scores=vec![json!(["[UNK]",0.0])];
            for i in 0..n {scores.push(json!([format!("t{i}"),-(i as f64+0.125)]));}
            json!({"type":kind,"vocab":scores,"unk_id":0,"byte_fallback":false})
        }
        _=>{
            let mut merges=Vec::new();
            vocab.insert("x".into(),json!(n+1));
            for i in 0..n {vocab.insert(format!("xt{i}"),json!(n+2+i));merges.push(json!(["x",format!("t{i}")]));}
            json!({"type":"BPE","vocab":vocab,"merges":merges,"unk_token":"[UNK]","fuse_unk":true,"byte_fallback":false,"ignore_merges":false})
        }
    };
    json!({"version":"1.0","model":model,"added_tokens":[],
        "normalizer":{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"String":"x"},"content":"y"}]},
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"}]},
        "decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
        "post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}
    })
}
fn time(mut f:impl FnMut()->Vec<u8>, count:usize)->f64 {
    (0..5).map(|_|{let start=Instant::now();for _ in 0..count {black_box(f());}start.elapsed().as_secs_f64()*1000.0}).fold(f64::INFINITY,f64::min)
}
fn main(){
    let mut cases=0;
    for kind in ["BPE","WordLevel","Unigram"] {for n in [16,1024,32768] {
        let bytes=serde_json::to_vec(&document(kind,n)).unwrap();
        let local=local::Tokenizer::from_bytes(&bytes).unwrap();
        let reference=reference::Tokenizer::from_bytes(&bytes).unwrap();
        let expected=serde_json::to_vec(&reference).unwrap();
        let meter=Meter(Cell::new(0));
        let paid=local::source_serialization::serialize(&local,&meter).unwrap();
        assert_eq!(paid,expected,"{kind}/{n} pristine bytes");
        assert_eq!(serde_json::to_vec(&local).unwrap(),expected);
        let count=if n<100 {2000}else if n<2000 {100}else{5};
        let pristine=time(||serde_json::to_vec(&reference).unwrap(),count);
        let ordinary=time(||serde_json::to_vec(&local).unwrap(),count);
        let paid=time(||local::source_serialization::serialize(&local,&meter).unwrap(),count);
        println!("{kind} vocab={n} bytes={} loops={count} pristine_ms={pristine:.3} ordinary_ms={ordinary:.3} paid_ms={paid:.3} ordinary_ratio={:.3} paid_ratio={:.3}",expected.len(),ordinary/pristine,paid/pristine);
        cases+=1;
    }}
    println!("PASS: {cases} complete source byte comparisons with pristine tokenizers 0.23.2");
}
