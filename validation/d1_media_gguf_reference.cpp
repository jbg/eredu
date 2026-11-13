// Same-artifact d1 media projection and single-question slot reference using llama.cpp.
#include <llama.h>
#include <mtmd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
mtmd::bitmap_ptr frame(int size, bool video, int index) {
    std::vector<unsigned char> rgb(3 * size * size);
    for (size_t i = 0; i < rgb.size(); i += 3) {
        rgb[i] = video ? 20 : 230;
        rgb[i + 1] = video ? 50 + 10 * index : 20;
        rgb[i + 2] = video ? 210 : 10;
    }
    mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(size, size, rgb.data()));
    if (!bitmap) throw std::runtime_error("construct RGB fixture");
    mtmd_bitmap_set_mergeable(bitmap.get(), video);
    mtmd_bitmap_set_is_video(bitmap.get(), video);
    return bitmap;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: BACKBONE PROJECTOR TOKENS_CSV MEDIA_ID image|video|audio OUTPUT_PREFIX");
        std::vector<llama_token> tokens;
        std::string ids = argv[3];
        while (!ids.empty()) {
            const auto comma = ids.find(',');
            tokens.push_back(std::stoi(ids.substr(0, comma)));
            ids = comma == std::string::npos ? "" : ids.substr(comma + 1);
        }
        const int marker = std::stoi(argv[4]);
        const std::string_view kind = argv[5];
        const std::string output = argv[6];
        if (kind != "image" && kind != "video" && kind != "audio") {
            throw std::runtime_error("unknown media fixture");
        }

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("load backbone");
        auto media_params = mtmd_context_params_default();
        media_params.use_gpu = false;
        media_params.warmup = false;
        media_params.n_threads = 8;
        media_params.image_min_tokens = 1;
        media_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> media(
            mtmd_init_from_file(argv[2], model.get(), media_params), mtmd_free);
        if (!media) throw std::runtime_error("load projector");
        std::vector<mtmd::bitmap_ptr> bitmaps;
        if (kind == "audio") {
            if (mtmd_get_audio_sample_rate(media.get()) != 16000) {
                throw std::runtime_error("expected a 16-kHz audio projector");
            }
            std::vector<float> pcm(2151);
            for (int i = 0; i < static_cast<int>(pcm.size()); ++i) {
                pcm[i] = ((i * 37) % 997 - 498) / 1024.0f;
            }
            bitmaps.emplace_back(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
        } else if (kind == "video") {
            for (int i = 0; i < 5; ++i) bitmaps.push_back(frame(96, true, i));
        } else {
            bitmaps.push_back(frame(256, false, 0));
        }
        std::vector<const mtmd_bitmap *> media_inputs;
        std::string prompt;
        for (const auto & bitmap : bitmaps) {
            if (!bitmap) throw std::runtime_error("construct media fixture");
            media_inputs.push_back(bitmap.get());
            prompt += mtmd_default_marker();
        }
        const mtmd_input_text input{prompt.data(), prompt.size(), false, true};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize(media.get(), chunks.get(), &input, media_inputs.data(), media_inputs.size())) {
            throw std::runtime_error("tokenize media");
        }
        const auto width = llama_model_n_embd_inp(model.get());
        std::vector<float> projected;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
            if (mtmd_encode_chunk(media.get(), chunk)) throw std::runtime_error("project media");
            const auto * rows = mtmd_get_output_embd(media.get());
            if (!rows) throw std::runtime_error("missing projected media rows");
            projected.insert(projected.end(), rows, rows + width * mtmd_input_chunk_get_n_tokens(chunk));
        }
        const auto media_tokens = static_cast<size_t>(std::count(tokens.begin(), tokens.end(), marker));
        if (projected.empty() || projected.size() != width * media_tokens) {
            throw std::runtime_error("media placeholder count mismatch " + std::to_string(projected.size() / width));
        }

        auto params = llama_context_default_params();
        params.n_ctx = tokens.size() + 8;
        params.n_batch = params.n_ubatch = tokens.size();
        params.n_threads = params.n_threads_batch = 8;
        params.type_k = params.type_v = GGML_TYPE_F32;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), params), llama_free);
        if (!context) throw std::runtime_error("initialize decoder");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(context.get()), llama_batch_ext_free);
        size_t media_row = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto pos = static_cast<llama_pos>(i);
            const auto idx = tokens[i] == marker
                ? llama_batch_ext_add_embd(batch.get(), 0, {projected.data() + width * media_row++, 1, size_t(width)})
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
                !llama_batch_ext_set_output_logits(batch.get(), idx, i + 1 == tokens.size())) {
                throw std::runtime_error("pack decoder row");
            }
        }
        if (llama_process(context.get(), LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("execute decoder row");
        }
        const auto * row = llama_get_logits_ith(context.get(), -1);
        if (!row) throw std::runtime_error("read answer logits");
        const auto save = [&](const std::string & name, const float * data, size_t size) {
            std::ofstream stream(output + name, std::ios::binary);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
            stream.write(reinterpret_cast<const char *>(data), size * sizeof(float));
        };
        save("-projection.f32", projected.data(), projected.size());
        save("-slot.f32", row, llama_vocab_n_tokens(llama_model_get_vocab(model.get())));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
