// Independent multimodal GGUF acceptance through llama.cpp and mtmd.
// Define EREDU_MTMD_VIDEO_API for the pinned mtmd API with temporal frame merging.
// Define EREDU_MTMD_AUDIO_API to enable mono float32 PCM input.
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void write_u64(std::ostream & output, uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        output.put(static_cast<char>((value >> (byte * 8)) & 255));
    }
}

static std::string json_string(const std::string & value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') output << '\\' << ch;
        else if (ch < 32) {
            constexpr char digits[] = "0123456789abcdef";
            output << "\\u00" << digits[ch >> 4] << digits[ch & 15];
        } else output << ch;
    }
    output << '"';
    return output.str();
}

static void write_tokens(std::ostream & output, const std::vector<llama_token> & values) {
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

#ifdef EREDU_MTMD_VIDEO_API
struct ClipFrames {
    int width;
    int height;
    double fps;
    const std::vector<unsigned char> & pixels;
    size_t frame_bytes;

    static int emit(size_t index, void * data, mtmd_bitmap ** bitmap, char ** text) {
        const auto & clip = *static_cast<const ClipFrames *>(data);
        const size_t count = clip.pixels.size() / clip.frame_bytes;
        const size_t first = index / 3 * 2;
        if (first >= count) return -1;
        const size_t last = std::min(first + 1, count - 1);
        if (index % 3 == 0) {
            char stamp[64];
            const int length = std::snprintf(stamp, sizeof(stamp), "<%.1f seconds>",
                                             (first / clip.fps + last / clip.fps) / 2);
            if (length < 0 || size_t(length) >= sizeof(stamp)) return -2;
            *text = static_cast<char *>(std::malloc(size_t(length) + 1));
            if (!*text) return -2;
            std::memcpy(*text, stamp, size_t(length) + 1);
        } else {
            const size_t frame = index % 3 == 1 ? first : last;
            *bitmap = mtmd_bitmap_init(clip.width, clip.height,
                                       clip.pixels.data() + frame * clip.frame_bytes);
            if (!*bitmap) return -2;
        }
        return 0;
    }
};

#endif

#ifdef EREDU_MTMD_AUDIO_API
static mtmd::bitmap_ptr read_audio(const char * path, int sample_rate, mtmd_context * media) {
    if (mtmd_get_audio_sample_rate(media) != sample_rate) {
        throw std::runtime_error("PCM sample rate differs from the projector sample rate");
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(float)) {
        throw std::runtime_error("expected complete nonempty float32 PCM");
    }
    file.seekg(0);
    std::vector<float> pcm(size_t(bytes) / sizeof(float));
    if (!file.read(reinterpret_cast<char *>(pcm.data()), bytes)) {
        throw std::runtime_error("cannot read PCM clip");
    }
    return mtmd::bitmap_ptr(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
}
#endif

int main(int argc, char ** argv) {
    try {
        if (argc < 8 || argc > 10) {
            throw std::runtime_error(
                "usage: MODEL PROJECTOR MEDIA WIDTH HEIGHT PROMPT OUTPUT [GREEDY_TOKENS [VIDEO_FPS]]\n"
                "for audio, replace WIDTH HEIGHT with --pcm-f32 SAMPLE_RATE");
        }
        const bool audio = std::string(argv[4]) == "--pcm-f32";
        const int width = audio ? 1 : std::stoi(argv[4]);
        const int height = std::stoi(argv[5]);
        if (width <= 0 || height <= 0) throw std::runtime_error("expected positive image dimensions or PCM sample rate");
        std::vector<llama_token> prompt;
        const int generated_tokens = argc >= 9 ? std::stoi(argv[8]) : 4;
        if (generated_tokens < 2 || generated_tokens > 128) {
            throw std::runtime_error("greedy token count must be between 2 and 128");
        }
        std::vector<llama_token> decode(generated_tokens - 1);
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        Model model(llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("llama.cpp could not load the artifact");
        const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        auto media_params = mtmd_context_params_default();
        media_params.use_gpu = false;
        media_params.warmup = false;
        media_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        media_params.n_threads = 8;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> media(
            mtmd_init_from_file(argv[2], model.get(), media_params), mtmd_free);
        if (!media) throw std::runtime_error("cannot open projector");
        size_t expected_media_chunks = 1;
        mtmd::bitmap_ptr bitmap;
        if (audio) {
#ifdef EREDU_MTMD_AUDIO_API
            if (argc >= 10) throw std::runtime_error("FPS is only supported for video");
            bitmap = read_audio(argv[3], height, media.get());
#else
            throw std::runtime_error("PCM input requires the mtmd audio API");
#endif
        } else {
            std::ifstream rgb_file(argv[3], std::ios::binary);
            const std::vector<unsigned char> rgb((std::istreambuf_iterator<char>(rgb_file)), {});
            const size_t frame_bytes = size_t(width) * size_t(height) * 3;
#ifdef EREDU_MTMD_VIDEO_API
            const double fps = argc >= 10 ? std::stod(argv[9]) : 0;
            ClipFrames clip{width, height, fps, rgb, frame_bytes};
            if (argc >= 10) {
                if (!std::isfinite(fps) || fps <= 0 || rgb.empty() || rgb.size() % frame_bytes != 0) {
                    throw std::runtime_error("video requires positive FPS and complete RGB frames");
                }
                expected_media_chunks = (rgb.size() / frame_bytes + 1) / 2;
                bitmap.reset(mtmd_bitmap_init_lazy(media.get(), nullptr, &clip, ClipFrames::emit));
            } else
#else
            if (argc >= 10) throw std::runtime_error("video requires the newer mtmd API");
#endif
            {
                if (rgb.size() != frame_bytes) {
                    throw std::runtime_error("RGB byte count differs from image dimensions");
                }
                bitmap.reset(mtmd_bitmap_init(width, height, rgb.data()));
            }
        }
        if (!bitmap) throw std::runtime_error("cannot create media input");
        const mtmd_bitmap * inputs[] = {bitmap.get()};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        std::ifstream prompt_file(argv[6], std::ios::binary);
        const std::string rendered((std::istreambuf_iterator<char>(prompt_file)), {});
        if (rendered.empty()) throw std::runtime_error("missing rendered prompt");
#ifdef EREDU_MTMD_VIDEO_API
        const mtmd_input_text input{rendered.c_str(), rendered.size(), false, true};
#else
        const mtmd_input_text input{rendered.c_str(), false, true};
#endif
        if (mtmd_tokenize(media.get(), chunks.get(), &input, inputs, 1))
            throw std::runtime_error("multimodal prompt tokenization failed");
        size_t media_chunks = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t length = 0;
                const auto * chunk_ids = mtmd_input_chunk_get_tokens_text(chunk, &length);
                if (length) prompt.insert(prompt.end(), chunk_ids, chunk_ids + length);
            } else {
                ++media_chunks;
                std::cerr << "media tokens=" << mtmd_input_chunk_get_n_tokens(chunk) << "\n";
            }
        }
        if (media_chunks != expected_media_chunks) {
            throw std::runtime_error("unexpected media chunk count");
        }
        auto context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;
        context_params.n_threads = 8;
        context_params.n_threads_batch = 8;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
        Context context(llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("llama.cpp could not create a context");
        std::vector<float> rows;
        llama_pos position = 0;
        if (mtmd_helper_eval_chunks(media.get(), context.get(), chunks.get(), 0, 0,
                                    context_params.n_batch, true, &position)) {
            throw std::runtime_error("multimodal prompt evaluation failed");
        }
        const auto append_scores = [&] {
            const float * row = llama_get_logits_ith(context.get(), -1);
            if (!row) throw std::runtime_error("missing vocabulary scores");
            rows.insert(rows.end(), row, row + vocabulary);
        };
        append_scores();
        for (auto & token : decode) {
            const auto begin = rows.end() - vocabulary;
            token = std::max_element(begin, rows.end()) - begin;
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = position++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            const auto status = llama_decode(context.get(), batch);
            llama_batch_free(batch);
            if (status) throw std::runtime_error("cached decode failed");
            append_scores();
        }
        const uint64_t first_bytes = uint64_t(vocabulary) * sizeof(float);
        const uint64_t total_bytes = uint64_t(rows.size()) * sizeof(float);
        std::ostringstream metadata;
        metadata << "{\"prefill.logits\":{\"dtype\":\"F32\",\"shape\":[1," << vocabulary
                 << "],\"data_offsets\":[0," << first_bytes
                 << "]},\"decode.logits\":{\"dtype\":\"F32\",\"shape\":[" << decode.size()
                 << "," << vocabulary << "],\"data_offsets\":[" << first_bytes << ","
                 << total_bytes << "]}}";
        auto header = metadata.str();
        header.append((8 - header.size() % 8) % 8, ' ');
        std::ofstream output(argv[7], std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        write_u64(output, header.size());
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) {
            throw std::runtime_error("reference exporter requires a little-endian host");
        }
        output.write(reinterpret_cast<const char *>(rows.data()), static_cast<std::streamsize>(total_bytes));
        output.close();
        auto report_path = std::filesystem::path(argv[7]);
        report_path.replace_extension(".json");
        std::ofstream report(report_path);
        report.exceptions(std::ios::failbit | std::ios::badbit);
        report << "{\"input\":{\"token_ids\":";
        write_tokens(report, prompt);
        report << "},\"output\":{\"fed_token_ids\":";
        write_tokens(report, decode);
        report << ",\"tensor_file\":" << json_string(std::filesystem::path(argv[7]).filename().string())
               << "},\"reference\":{\"implementation\":\"llama.cpp\",\"device\":\"cpu\""
               << ",\"kv_dtype\":\"float32\",\"flash_attention\":false}}\n";
        report.close();
        context.reset();
        media.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
