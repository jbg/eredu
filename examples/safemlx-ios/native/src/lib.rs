//! Small C ABI used by the SafeMLX iOS example.

use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::PathBuf,
    ptr,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use eredu::{
    api::LoadedModel, backend::mlx::MlxBackend, runtime::chat::ChatTemplateRequest,
    GenerationConfigOverrides, TextGenerationConfig, TokenOutput,
};
use safemlx::{Device, DeviceType, ExecutionContext};

/// Receives one UTF-8 text fragment. The bytes are valid only during the call.
pub type TextCallback = unsafe extern "C" fn(*const u8, usize, *mut c_void);

#[derive(Debug, Clone, Copy, Default)]
struct GenerationStats {
    generated_tokens: u64,
    ttft_seconds: f64,
    tokens_per_second: f64,
}

impl GenerationStats {
    fn new(generated_tokens: u64, ttft: Option<Duration>, elapsed: Duration) -> Self {
        let ttft = ttft.unwrap_or(elapsed);
        let decode_tokens = generated_tokens.saturating_sub(1);
        let decode_seconds = elapsed.saturating_sub(ttft).as_secs_f64();
        Self {
            generated_tokens,
            ttft_seconds: ttft.as_secs_f64(),
            tokens_per_second: if decode_tokens > 0 && decode_seconds > 0.0 {
                decode_tokens as f64 / decode_seconds
            } else {
                0.0
            },
        }
    }
}

enum Command {
    Generate {
        prompt: String,
        callback: TextCallback,
        context: usize,
        result: Sender<Result<GenerationStats, String>>,
    },
    Shutdown,
}

/// Opaque, thread-safe command endpoint. The MLX objects never leave `worker`.
pub struct ModelHandle {
    commands: Sender<Command>,
    worker: Option<JoinHandle<()>>,
}

fn c_string(value: *const c_char, label: &str) -> Result<String, String> {
    if value.is_null() {
        return Err(format!("{label} must not be null"));
    }
    // SAFETY: the C ABI requires a live, NUL-terminated string for the call.
    let value = unsafe { CStr::from_ptr(value) };
    value
        .to_str()
        .map(str::to_owned)
        .map_err(|_| format!("{label} must be UTF-8"))
}

fn publish_error(out: *mut *mut c_char, message: String) {
    if out.is_null() {
        return;
    }
    let sanitized = message.replace('\0', "�");
    let message = CString::new(sanitized).expect("NUL bytes were replaced");
    // SAFETY: `out` is caller-provided writable storage. Ownership of the
    // allocation passes to the caller, which releases it with `safemlx_string_free`.
    unsafe { *out = message.into_raw() };
}

fn generate(
    model: &mut LoadedModel<MlxBackend<'static>>,
    prompt: &str,
    callback: TextCallback,
    context: usize,
) -> Result<GenerationStats, String> {
    let started = Instant::now();
    let rendered = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({
                "role": "user",
                "content": prompt,
            })],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        })
        .map(|prepared| (prepared.rendered_prompt().to_owned(), false))
        .or_else(|error| {
            if matches!(error, eredu::api::TextModelError::MissingChatTemplate) {
                Ok((prompt.to_owned(), true))
            } else {
                Err(error)
            }
        })
        .map_err(|error| error.to_string())?;

    let tokens = model
        .encode(&rendered.0, rendered.1)
        .map_err(|error| error.to_string())?;
    if tokens.is_empty() {
        return Err("the prompt produced no input tokens".into());
    }

    let mut settings = model
        .resolve_generation_config(GenerationConfigOverrides::default())
        .map_err(|error| error.to_string())?;
    let max_tokens = settings.max_new_tokens.unwrap_or(256);
    settings.max_new_tokens = Some(max_tokens);
    let eos = model.eos_token_ids().to_vec();
    let mut decoder = model.text_decoder(true);
    let generator = model
        .generate_tokens(tokens, TextGenerationConfig::new(settings))
        .map_err(|error| error.to_string())?;
    let mut generated_tokens = 0_u64;
    let mut ttft = None;
    for token in generator {
        let token_id = token
            .map_err(|error| error.to_string())?
            .token_id()
            .map_err(|error| error.to_string())?;
        if eos.contains(&token_id) {
            break;
        }
        generated_tokens += 1;
        if ttft.is_none() {
            ttft = Some(started.elapsed());
        }
        if let Some(fragment) = decoder.step(token_id).map_err(|error| error.to_string())? {
            // SAFETY: callback and context originate from the synchronous C call,
            // which remains active until this worker reports completion.
            unsafe { callback(fragment.as_ptr(), fragment.len(), context as *mut c_void) };
        }
    }
    Ok(GenerationStats::new(
        generated_tokens,
        ttft,
        started.elapsed(),
    ))
}

fn worker_main(
    model_path: PathBuf,
    metallib_path: PathBuf,
    commands: mpsc::Receiver<Command>,
    ready: Sender<Result<(), String>>,
) {
    let initialized = (|| {
        safemlx::metal::set_metallib_path(&metallib_path).map_err(|error| error.to_string())?;
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model = LoadedModel::load(
            MlxBackend::new(execution.stream(), weights.stream()),
            &model_path,
            Default::default(),
        )
        .map_err(|error| error.to_string())?;
        execution
            .stream()
            .synchronize()
            .map_err(|error| error.to_string())?;
        ready
            .send(Ok(()))
            .map_err(|_| "loader disconnected".to_string())?;

        while let Ok(command) = commands.recv() {
            match command {
                Command::Generate {
                    prompt,
                    callback,
                    context,
                    result,
                } => {
                    let generated = generate(&mut model, &prompt, callback, context);
                    let _ = result.send(generated);
                }
                Command::Shutdown => break,
            }
        }
        Ok(())
    })();

    if let Err(error) = initialized {
        let _ = ready.send(Err(error));
    }
}

/// Loads a model on a dedicated native thread.
///
/// Returns null and sets `error_out` on failure.
#[no_mangle]
pub extern "C" fn safemlx_model_create(
    model_path: *const c_char,
    metallib_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut ModelHandle {
    if !error_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *error_out = ptr::null_mut() };
    }
    let result = (|| {
        let model_path = PathBuf::from(c_string(model_path, "model_path")?);
        let metallib_path = PathBuf::from(c_string(metallib_path, "metallib_path")?);
        let (commands, receiver) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("safemlx-model".into())
            .spawn(move || worker_main(model_path, metallib_path, receiver, ready_sender))
            .map_err(|error| error.to_string())?;
        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Box::new(ModelHandle {
                commands,
                worker: Some(worker),
            })),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err("model loader terminated unexpectedly".into())
            }
        }
    })();

    match result {
        Ok(handle) => Box::into_raw(handle),
        Err(error) => {
            publish_error(error_out, error);
            ptr::null_mut()
        }
    }
}

/// Generates one response using checkpoint defaults and streams UTF-8 fragments.
///
/// Returns zero on success and sets `error_out` on failure.
#[no_mangle]
pub extern "C" fn safemlx_model_generate(
    handle: *mut ModelHandle,
    prompt: *const c_char,
    callback: Option<TextCallback>,
    context: *mut c_void,
    generated_tokens_out: *mut u64,
    ttft_seconds_out: *mut f64,
    tokens_per_second_out: *mut f64,
    error_out: *mut *mut c_char,
) -> i32 {
    if !error_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *error_out = ptr::null_mut() };
    }
    if !generated_tokens_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *generated_tokens_out = 0 };
    }
    if !ttft_seconds_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *ttft_seconds_out = 0.0 };
    }
    if !tokens_per_second_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *tokens_per_second_out = 0.0 };
    }
    let result = (|| {
        if handle.is_null() {
            return Err("model handle must not be null".into());
        }
        let callback = callback.ok_or_else(|| "text callback must not be null".to_string())?;
        let prompt = c_string(prompt, "prompt")?;
        let (sender, receiver) = mpsc::channel();
        // SAFETY: the handle is owned by Swift until this synchronous call returns.
        let handle = unsafe { &*handle };
        handle
            .commands
            .send(Command::Generate {
                prompt,
                callback,
                context: context as usize,
                result: sender,
            })
            .map_err(|_| "model worker is unavailable".to_string())?;
        receiver
            .recv()
            .map_err(|_| "model worker terminated during generation".to_string())?
    })();
    match result {
        Ok(stats) => {
            if !generated_tokens_out.is_null() {
                // SAFETY: checked non-null writable out parameter.
                unsafe { *generated_tokens_out = stats.generated_tokens };
            }
            if !ttft_seconds_out.is_null() {
                // SAFETY: checked non-null writable out parameter.
                unsafe { *ttft_seconds_out = stats.ttft_seconds };
            }
            if !tokens_per_second_out.is_null() {
                // SAFETY: checked non-null writable out parameter.
                unsafe { *tokens_per_second_out = stats.tokens_per_second };
            }
            0
        }
        Err(error) => {
            publish_error(error_out, error);
            1
        }
    }
}

/// Stops the worker and releases a model handle.
#[no_mangle]
pub extern "C" fn safemlx_model_free(handle: *mut ModelHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: ownership was created by `safemlx_model_create` and is returned once.
    let mut handle = unsafe { Box::from_raw(handle) };
    let _ = handle.commands.send(Command::Shutdown);
    if let Some(worker) = handle.worker.take() {
        let _ = worker.join();
    }
}

/// Releases an error string returned by this library.
#[no_mangle]
pub extern "C" fn safemlx_string_free(value: *mut c_char) {
    if !value.is_null() {
        // SAFETY: ownership was created with `CString::into_raw` by this library.
        drop(unsafe { CString::from_raw(value) });
    }
}

#[cfg(test)]
mod tests {
    use super::GenerationStats;
    use std::time::Duration;

    #[test]
    fn generation_stats_exclude_ttft_and_first_token_from_decode_rate() {
        let stats = GenerationStats::new(
            10,
            Some(Duration::from_millis(500)),
            Duration::from_millis(1400),
        );
        assert_eq!(stats.generated_tokens, 10);
        assert_eq!(stats.ttft_seconds, 0.5);
        assert!((stats.tokens_per_second - 10.0).abs() < f64::EPSILON);
    }
}
