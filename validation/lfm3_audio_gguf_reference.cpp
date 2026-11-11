// Mixed audio prompt and speech feedback scores from the pinned llama.cpp C API.
#include <llama.h>
#include <mtmd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

static void depth_reference(mtmd_context * media, const char * path, int width) {
    if (width != 2048) throw std::runtime_error("unexpected Audio backbone width");
    constexpr int books = 12;
    constexpr int vocabulary = 2017;
    constexpr int order[books] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    std::array<int32_t, books> codes;
    codes.fill(-1);
    std::vector<float> hidden(width);
    for (int i = 0; i < width; ++i) hidden[i] = float(i % 67 - 33) / 32.0f;
    mtmd::gen_state_ptr state(mtmd_gen_state_init(media, mtmd_gen_state_params_default()));
    if (!state) throw std::runtime_error("cannot initialize depth decoder");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    for (int pass = 0; pass < 3; ++pass) {
        auto input = mtmd_gen_inp_default(media);
        input.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        input.embd = hidden.data();
        input.codes = codes.data();
        input.n_codes = codes.size();
        mtmd_gen_out result{};
        if (mtmd_gen_audio_process(media, state.get(), &input, &result) ||
            !result.feats || result.n_feats != books * vocabulary) {
            throw std::runtime_error("missing full depth score rows");
        }
        out.write(reinterpret_cast<const char *>(result.feats), result.n_feats * sizeof(float));
        for (int i = pass * 4; i < (pass + 1) * 4; ++i) {
            const auto * scores = result.feats + order[i] * vocabulary;
            codes[order[i]] = int32_t(std::max_element(scores, scores + vocabulary - 1) - scores);
        }
    }
}

int main(int argc, char ** argv) {
    try {
        const bool depth = argc == 5 && std::string_view(argv[4]) == "--depth";
        if (argc != 4 && !depth) {
            throw std::runtime_error("usage: audio-reference BACKBONE.gguf PROJECTOR.gguf SCORES.f32 [--depth]");
        }
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("cannot open backbone");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto media_params = mtmd_context_params_default();
        media_params.use_gpu = true;
        media_params.warmup = false;
        media_params.n_threads = 4;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> media(
            mtmd_init_from_file(argv[2], model.get(), media_params), mtmd_free);
        if (!media) throw std::runtime_error("cannot initialize media context");
        if (depth) {
            depth_reference(media.get(), argv[3], llama_model_n_embd_inp(model.get()));
            return 0;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("cannot initialize text context");

        std::vector<float> pcm(2151);
        constexpr float tau = 6.2831853071795864769f;
        for (size_t i = 0; i < pcm.size(); ++i) {
            const float time = float(i) / 16000.0f;
            pcm[i] = 0.12f * std::sin(tau * 413.0f * time) + 0.05f * std::sin(tau * 863.0f * time);
        }
        mtmd::bitmap_ptr bitmap(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        const std::string_view marker(mtmd_default_marker());
        const mtmd_input_text text{marker.data(), marker.size(), false, true};
        const mtmd_bitmap * input_bitmap = bitmap.get();
        if (!bitmap || mtmd_tokenize(media.get(), chunks.get(), &text, &input_bitmap, 1)) {
            throw std::runtime_error("audio preprocessing failed");
        }
        const mtmd_input_chunk * audio = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) audio = chunk;
        }
        if (!audio || mtmd_input_chunk_get_n_tokens(audio) != 2 || mtmd_encode_chunk(media.get(), audio)) {
            throw std::runtime_error("expected two encoded audio rows");
        }
        const int width = llama_model_n_embd_inp(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        int position = 0;
        const auto decode = [&](llama_token * tokens, const float * embeddings, int count, bool scores) {
            auto batch = llama_batch_init(count, embeddings ? width : 0, 1);
            batch.n_tokens = count;
            if (tokens) std::copy_n(tokens, count, batch.token);
            else std::copy_n(embeddings, count * width, batch.embd);
            for (int i = 0; i < count; ++i) {
                batch.pos[i] = position++;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = scores && i == count - 1;
            }
            const int result = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (result != 0) throw std::runtime_error("llama_decode failed");
        };

        std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        const auto save_scores = [&] {
            const auto * row = llama_get_logits_ith(ctx.get(), -1);
            if (!row) throw std::runtime_error("missing score row");
            out.write(reinterpret_cast<const char *>(row), vocab * sizeof(float));
        };
        llama_token prefix[] = {124894, 5};
        llama_token suffix[] = {13, 21, 34};
        decode(prefix, nullptr, 2, false);
        decode(nullptr, mtmd_get_output_embd(media.get()), 2, false);
        decode(suffix, nullptr, 3, true);
        save_scores();

        mtmd::gen_state_ptr gen(mtmd_gen_state_init(media.get(), mtmd_gen_state_params_default()));
        auto input = mtmd_gen_inp_default(media.get());
        input.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        std::array<int32_t, 12> codes{};
        for (int book = 0; book < 12; ++book) codes[book] = 149 * book;
        input.codes = codes.data();
        input.n_codes = codes.size();
        mtmd_gen_out output{};
        if (!gen || mtmd_gen_audio_process(media.get(), gen.get(), &input, &output) || !output.embd) {
            throw std::runtime_error("speech feedback projection failed");
        }
        decode(nullptr, output.embd, 1, true);
        save_scores();
        for (llama_token token : {5, 8, 13, 21, 34, 55, 3}) {
            decode(&token, nullptr, 1, true);
            save_scores();
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
