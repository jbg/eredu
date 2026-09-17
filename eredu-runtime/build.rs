//! Qualify the reviewed fresh-Vec, concrete fresh-clone and Arc/Rc producers.
//! Their exact liballoc bodies share these compiler/library pins; this does not
//! qualify allocator-private bookkeeping or a total-process memory guarantee.
//! Fresh scalar Vec clones use slice::to_vec_in(len); String delegates to its
//! byte Vec; pinned Darwin PathBuf/OsString/Buf delegate to that same byte Vec.
//! Arbitrary Clone, clone_from and iterator growth are not qualified.
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

fn hash(path: &Path) -> Option<String> {
    println!("cargo:rerun-if-changed={}", path.display());
    let mut file = fs::File::open(path).ok()?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).ok()?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Some(
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}
fn output(compiler: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new(compiler).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}
fn compiler_path() -> Option<PathBuf> {
    let compiler = PathBuf::from(env::var_os("RUSTC")?);
    if compiler.is_absolute() {
        // Track the spelling and its directory as well as the canonical target:
        // replacing an executable symlink must invalidate qualification.
        println!("cargo:rerun-if-changed={}", compiler.display());
        println!("cargo:rerun-if-changed={}", compiler.parent()?.display());
        return compiler.canonicalize().ok();
    }
    // Cargo can pass a bare command name. Resolve it through absolute PATH
    // entries, as process execution does, before hashing the actual executable.
    // Relative paths depend on Cargo/build-script working directories and are
    // deliberately outside this qualification.
    if compiler.components().count() != 1 {
        return None;
    }
    let search = env::var_os("PATH")?;
    let paths = env::split_paths(&search).collect::<Vec<_>>();
    if paths.iter().any(|path| !path.is_absolute()) {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for path in paths {
            // Watching searched directories detects a newly created earlier
            // executable and replacement of the selected symlink. Do not watch
            // nonexistent candidates: Cargo would rerun on every invocation.
            // Missing search directories remain conservatively unqualified.
            if !path.is_dir() {
                return None;
            }
            println!("cargo:rerun-if-changed={}", path.display());
            let candidate = path.join(&compiler);
            if fs::metadata(&candidate).is_ok_and(|metadata| {
                metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
            }) {
                return candidate.canonicalize().ok();
            }
        }
    }
    None
}
fn qualified() -> Option<()> {
    if env::var("TARGET").ok()?.as_str() != "aarch64-apple-darwin" {
        return None;
    }
    for name in [
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_BOOTSTRAP",
    ] {
        if env::var_os(name).is_some_and(|s| !s.is_empty()) {
            return None;
        }
    }
    // A replacement sysroot/extern/std build is a different producer, even if
    // its driver prints the same release. Unknown overrides stay unqualified.
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] {
        let flags = env::var(name).unwrap_or_default();
        // Response files can hide every override below. Dynamic linking selects
        // an unpinned std artifact. Neither route has this producer qualification.
        if flags.contains('@') || flags.contains("prefer-dynamic") {
            return None;
        }
        if ["--sysroot", "--extern", "-Z", "-L"]
            .iter()
            .any(|flag| flags.contains(flag))
        {
            return None;
        }
    }
    let compiler = compiler_path()?;
    if hash(&compiler)? != "a11618eca0956a8aa4372c2bc898690b513cbdfa2cb9125b2a5301e360ed5b49" {
        return None;
    }
    let version = output(&compiler, &["-vV"])?;
    if !version.lines().any(|l| l == "release: 1.98.0")
        || !version
            .lines()
            .any(|l| l.starts_with("commit-hash: 88d9e12ae"))
    {
        return None;
    }
    let sysroot = PathBuf::from(output(&compiler, &["--print", "sysroot"])?.trim());
    if hash(&sysroot.join("lib/librustc_driver-4031c0ff8e88f5d1.dylib"))?
        != "275171d3d528b7f78bcad812a84658ec3bd756ce9792a329b6edefdf70884c63"
    {
        return None;
    }
    let libraries = PathBuf::from(
        output(
            &compiler,
            &[
                "--print",
                "target-libdir",
                "--target",
                "aarch64-apple-darwin",
            ],
        )?
        .trim(),
    );
    for (name, expected) in [
        (
            "liballoc-4804d8e0d239d123.rlib",
            "09d81058d3f6962fad6672db5acd84cc893d1bf62535b3592529b0bc3e3b1159",
        ),
        (
            "libstd-61f27fb94867ea3a.rlib",
            "ac6a7b8a0830f05b2228b5ca737aa49e6559c85838fe4e774e926d689ddb28b4",
        ),
    ] {
        if hash(&libraries.join(name))? != expected {
            return None;
        }
    }
    Some(())
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for name in [
        "TARGET",
        "RUSTC",
        "PATH",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_BOOTSTRAP",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let known = qualified().is_some();
    let generated = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"))
        .join("qualified_fresh_vec.rs");
    // A user --cfg flag cannot manufacture this private producer fact.
    fs::write(
        generated,
        format!("const QUALIFIED_FRESH_VEC: bool = {known};\n"),
    )
    .expect("write producer qualification");
}
