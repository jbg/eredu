//! Small C ABI used by the SafeMLX iOS example.

use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::PathBuf,
    ptr,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
};

use safemlx::{transforms::async_eval, Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    api::{GenerationConfigOverrides, LoadedModel},
    runtime::{
        chat::ChatTemplateRequest,
        media::input::{InputPart, ModelInput},
    },
};

/// Receives one UTF-8 text fragment. The bytes are valid only during the call.
pub type TextCallback = unsafe extern "C" fn(*const u8, usize, *mut c_void);

enum Command {
    Generate {
        prompt: String,
        callback: TextCallback,
        context: usize,
        result: Sender<Result<(), String>>,
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
    model: &mut LoadedModel,
    stream: &safemlx::Stream,
    prompt: &str,
    callback: TextCallback,
    context: usize,
) -> Result<(), String> {
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
            if matches!(error, safemlx_lm::error::Error::MissingChatTemplate) {
                Ok((prompt.to_owned(), true))
            } else {
                Err(error)
            }
        })
        .map_err(|error| error.to_string())?;

    let tokens = model
        .encode_to_array(&rendered.0, rendered.1, stream)
        .map_err(|error| error.to_string())?;
    if tokens.shape().get(1).copied().unwrap_or_default() == 0 {
        return Err("the prompt produced no input tokens".into());
    }

    let settings = model
        .resolve_generation_config(GenerationConfigOverrides::default())
        .map_err(|error| error.to_string())?;
    let max_tokens = settings.max_new_tokens.unwrap_or(256);
    let prng_key = (settings.temperature != 0.0)
        .then(|| safemlx::random::key(0))
        .transpose()
        .map_err(|error| error.to_string())?;
    let eos = model.eos_token_ids().to_vec();
    let mut cache = model.new_cache();
    let mut decoder = model.text_decoder(true);
    let parts = [InputPart::text_token_ids(&tokens)];
    let input = ModelInput::new(&parts);
    let mut generator = model.generate_input_with_cache_sampler(
        &mut cache,
        settings.temperature,
        input,
        prng_key,
        stream,
        settings.sampler(),
    );

    let mut current = generator
        .next()
        .transpose()
        .map_err(|error| error.to_string())?;
    for index in 0..max_tokens {
        let Some(token) = current.take() else { break };
        let next = if index + 1 < max_tokens {
            let next = generator.next();
            if let Some(Ok(next_token)) = next.as_ref() {
                async_eval([next_token]).map_err(|error| error.to_string())?;
            }
            next
        } else {
            None
        };
        let token_id = token.item::<u32>(stream);
        if eos.contains(&token_id) {
            break;
        }
        if let Some(fragment) = decoder.step(token_id).map_err(|error| error.to_string())? {
            // SAFETY: callback and context originate from the synchronous C call,
            // which remains active until this worker reports completion.
            unsafe { callback(fragment.as_ptr(), fragment.len(), context as *mut c_void) };
        }
        current = next.transpose().map_err(|error| error.to_string())?;
    }
    stream.synchronize().map_err(|error| error.to_string())
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
        let mut model = LoadedModel::load(&model_path, execution.stream(), weights.stream())
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
                    let generated =
                        generate(&mut model, execution.stream(), &prompt, callback, context);
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
    error_out: *mut *mut c_char,
) -> i32 {
    if !error_out.is_null() {
        // SAFETY: checked non-null writable out parameter.
        unsafe { *error_out = ptr::null_mut() };
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
        Ok(()) => 0,
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
