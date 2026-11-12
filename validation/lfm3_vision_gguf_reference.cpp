// Pre-sized LFM3 VL projection and conditional score reference using llama.cpp.
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
constexpr int width = 96;
constexpr int height = 64;
constexpr int media_rows = 6;
constexpr int cached_steps = 8;

mtmd::bitmap_ptr frame(int index, bool video) {
    std::vector<unsigned char> rgb(width * height * 3);
    for (int pixel = 0; pixel < width * height; ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            rgb[3 * pixel + channel] =
                static_cast<unsigned char>(17 * ((index * 3 + channel) * width * height + pixel));
        }
    }
    mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(width, height, rgb.data()));
    if (!bitmap) throw std::runtime_error("cannot construct RGB input");
    mtmd_bitmap_set_mergeable(bitmap.get(), video);
    mtmd_bitmap_set_is_video(bitmap.get(), video);
    return bitmap;
}

std::vector<float> project(mtmd_context * ctx, const std::vector<const mtmd_bitmap *> & frames,
                           int embedding_width) {
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    std::string prompt;
    for (size_t i = 0; i < frames.size(); ++i) prompt += mtmd_default_marker();
    const mtmd_input_text input{prompt.data(), prompt.size(), false, true};
    if (mtmd_tokenize(ctx, chunks.get(), &input, frames.data(), frames.size())) {
        throw std::runtime_error("cannot tokenize pre-sized media");
    }
    std::vector<float> result;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
        auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_input_chunk_get_n_tokens(chunk) != media_rows || mtmd_encode_chunk(ctx, chunk)) {
            throw std::runtime_error("unexpected media geometry or projection failure");
        }
        const float * rows = mtmd_get_output_embd(ctx);
        if (!rows) throw std::runtime_error("missing projected media rows");
        result.insert(result.end(), rows, rows + media_rows * embedding_width);
    }
    const size_t expected_groups = (frames.size() + 1) / 2;
    if (result.size() != expected_groups * media_rows * embedding_width) {
        throw std::runtime_error("unexpected temporal grouping");
    }
    return result;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4) {
            throw std::runtime_error(
                "usage: vision-reference BACKBONE.gguf PROJECTOR.gguf ROWS.f32 [--single-row] [--projector-cpu]");
        }
        bool single_row = false;
        bool projector_cpu = false;
        for (int i = 4; i < argc; ++i) {
            const std::string_view option(argv[i]);
            if (option == "--single-row") single_row = true;
            else if (option == "--projector-cpu") projector_cpu = true;
            else throw std::runtime_error("unknown reference option");
        }
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("cannot open backbone");
        auto media_params = mtmd_context_params_default();
        media_params.use_gpu = !projector_cpu;
        media_params.warmup = false;
        media_params.image_min_tokens = 1;
        media_params.n_threads = 4;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> media(
            mtmd_init_from_file(argv[2], model.get(), media_params), mtmd_free);
        if (!media) throw std::runtime_error("cannot open projector");

        const int embedding_width = llama_model_n_embd_inp(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const auto image = frame(0, false);
        const auto first = frame(0, true);
        const auto second = frame(1, true);
        const auto third = frame(2, true);
        const auto image_projection = project(media.get(), {image.get()}, embedding_width);
        const auto video_projection = project(media.get(), {first.get(), second.get(), third.get()},
                                              embedding_width);
        std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        const auto save = [&](const float * values, size_t size) {
            out.write(reinterpret_cast<const char *>(values), size * sizeof(float));
        };
        save(image_projection.data(), image_projection.size());
        save(video_projection.data(), video_projection.size());

        auto context_params = llama_context_default_params();
        context_params.n_ctx = 128;
        context_params.n_batch = 32;
        context_params.n_ubatch = single_row ? 1 : 32;
        context_params.type_k = context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("cannot initialize decoder");
        int position = 0;
        const auto decode = [&](const llama_token * tokens, const float * embeddings, int count) {
            auto batch = llama_batch_init(count, embeddings ? embedding_width : 0, 1);
            batch.n_tokens = count;
            if (tokens) std::copy_n(tokens, count, batch.token);
            else std::copy_n(embeddings, count * embedding_width, batch.embd);
            for (int i = 0; i < count; ++i) {
                batch.pos[i] = position++;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            const int status = llama_decode(context.get(), batch);
            llama_batch_free(batch);
            if (status) throw std::runtime_error("llama_decode failed");
            for (int i = 0; i < count; ++i) {
                const auto * row = llama_get_logits_ith(context.get(), i);
                if (!row) throw std::runtime_error("missing full vocabulary score row");
                save(row, vocab);
            }
        };
        const llama_token prefix[] = {124894, 5, 125009};
        const llama_token first_gap[] = {125010, 8, 125009};
        const llama_token second_gap[] = {125010, 13, 125009};
        const llama_token suffix[] = {125010, 21, 34};
        decode(prefix, nullptr, 3);
        decode(nullptr, image_projection.data(), media_rows);
        decode(first_gap, nullptr, 3);
        decode(nullptr, video_projection.data(), media_rows);
        decode(second_gap, nullptr, 3);
        decode(nullptr, video_projection.data() + media_rows * embedding_width, media_rows);
        decode(suffix, nullptr, 3);
        for (int i = 0; i < cached_steps; ++i) {
            const auto * row = llama_get_logits_ith(context.get(), -1);
            const llama_token token = std::max_element(row, row + vocab) - row;
            std::cout << token << '\n';
            decode(&token, nullptr, 1);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
