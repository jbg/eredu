//! Released tokenizer profiling without allocating model weights or a native device.
use super::*;

#[test]
#[ignore = "requires pinned DiffusionGemma sidecars in EREDU_DIFFUSION_TOKENIZER_DIR"]
fn released_diffusion_gemma_chat_preparation() {
    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_DIFFUSION_TOKENIZER_DIR")
            .expect("set EREDU_DIFFUSION_TOKENIZER_DIR"),
    );
    // Pin the sidecars used for the reproducible preparation measurement.
    use sha2::{Digest, Sha256};
    for (name, digest) in [
        (
            "tokenizer.json",
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        ),
        (
            "tokenizer_config.json",
            "a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200",
        ),
        (
            "chat_template.jinja",
            "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf",
        ),
        (
            "config.json",
            "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506",
        ),
        (
            "generation_config.json",
            "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
        ),
    ] {
        assert_eq!(
            Sha256::digest(fs::read(root.join(name)).unwrap())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            digest,
            "sidecar revision differs: {name}"
        );
    }
    let started = std::time::Instant::now();
    let raw = load_tokenizer(&root).unwrap();
    let eos = eos_token_ids_from_sidecar_dir(&root).unwrap();
    let template = load_chat_template(&root).unwrap().unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(&root).unwrap());
    eprintln!(
        "tokenizer_load_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    let started = std::time::Instant::now();
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos);
    eprintln!("compiler_seconds={:.6}", started.elapsed().as_secs_f64());
    let request = ChatTemplateRequest {
        messages: vec![
            json!({"role":"user","content":"Use the supplied evidence to answer each question. Select exactly one listed letter for each slot. Reply only with lines q0: A, q1: B, and so on, using the selected letters. Evidence and questions:\n{\"questions\":[{\"alternatives\":[{\"code\":\"A\",\"label\":\"yes\"},{\"code\":\"B\",\"label\":\"no\"}],\"question\":\"Is a refund appropriate?\",\"slot\":\"q0\"}],\"state\":\"duplicate charge\"}"}),
        ],
        enable_thinking: Some(false),
        add_generation_prompt: true,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "google/diffusiongemma-26B-A4B-it",
        &eos,
        Some(&compiler),
        request,
    )
    .unwrap();
    eprintln!(
        "semantic_prepare_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    let started = std::time::Instant::now();
    let ids = tokenizer.encode(prepared.rendered_prompt(), false).unwrap();
    assert_eq!(ids.len(), 97);
    eprintln!(
        "encode_seconds={:.6} prompt_tokens={}",
        started.elapsed().as_secs_f64(),
        ids.len()
    );
}
