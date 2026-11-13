// Build against the pinned llama.cpp mtmd library. Writes mono F32 PCM.
#include "llama.h"
#include "mtmd.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4 && !(argc == 5 && std::string_view(argv[4]) == "--metal")) {
        std::cerr << "usage: liquid-reference text.gguf projector.gguf output.f32 [--metal]\n";
        return 2;
    }

    llama_backend_init();
    auto text_params = llama_model_default_params();
    text_params.vocab_only = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> text(
        llama_model_load_from_file(argv[1], text_params), llama_model_free);
    if (!text) {
        return 1;
    }

    auto params = mtmd_context_params_default();
    params.use_gpu = argc == 5;
    params.warmup = false;
    params.n_threads = 4;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> ctx(
        mtmd_init_from_file(argv[2], text.get(), params), mtmd_free);
    if (!ctx) {
        return 1;
    }
    std::unique_ptr<mtmd_gen_state, decltype(&mtmd_gen_state_free)> state(
        mtmd_gen_state_init(ctx.get(), mtmd_gen_state_params_default()), mtmd_gen_state_free);
    if (!state) {
        return 1;
    }

    std::ofstream output(argv[3], std::ios::binary);
    int frame = 0;
    const std::array<int, 4> chunk_frames{1, 3, 6, 0};
    const std::array<std::size_t, 4> expected_samples{1440, 5760, 11520, 480};
    for (std::size_t chunk = 0; chunk < chunk_frames.size(); ++chunk) {
        std::vector<int32_t> codes;
        for (int end = frame + chunk_frames[chunk]; frame < end; ++frame) {
            for (int book = 0; book < 12; ++book) {
                codes.push_back((73 * frame + 149 * book + 31 * frame * book) % 2016);
            }
        }
        auto input = mtmd_gen_inp_default(ctx.get());
        input.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        input.codes = codes.data();
        input.n_codes = codes.size();
        mtmd_gen_out result{};
        if (mtmd_gen_audio_process(ctx.get(), state.get(), &input, &result) != 0 ||
            result.n_samples != expected_samples[chunk]) {
            std::cerr << "unexpected output at chunk " << chunk << '\n';
            return 1;
        }
        output.write(reinterpret_cast<const char *>(result.audio),
                     result.n_samples * sizeof(float));
    }
    return output ? 0 : 1;
}
