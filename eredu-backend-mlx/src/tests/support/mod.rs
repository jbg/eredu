#[allow(dead_code)]
pub(crate) mod expert_dispatch;
pub(crate) mod grouped_provider;
pub(crate) mod path_instrumentation;
pub(crate) mod test_utils;

pub(crate) fn read_prompt_cache(
    path: impl AsRef<std::path::Path>,
) -> Box<dyn eredu_core::cache::PromptCacheRead> {
    use eredu_core::cache::PromptCacheStore;
    eredu_runtime::DirectoryPromptCacheStore::new(path.as_ref(), u64::MAX)
        .open()
        .unwrap()
        .unwrap()
}
pub(crate) fn cache_cpu_stream() -> safemlx::Stream {
    safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}
