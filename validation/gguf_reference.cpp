// Independent full-vocabulary GGUF acceptance through llama.cpp's public C API.
// Build against the exact llama.cpp revision recorded in the acceptance manifest.
#include <llama.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<llama_token> tokens(const char * text) {
    std::vector<llama_token> result;
    std::istringstream input(text);
    std::string word;
    while (std::getline(input, word, ',')) {
        size_t used = 0;
        const long value = std::stol(word, &used);
        if (used != word.size() || value < 0 || value > INT32_MAX) {
            throw std::runtime_error("invalid token ID");
        }
        result.push_back(static_cast<llama_token>(value));
    }
    return result;
}

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

int main(int argc, char ** argv) {
    try {
        if (argc < 5 || argc > 7 || (argc == 7 && std::string(argv[6]) != "--single-row")) {
            throw std::runtime_error("usage: gguf_reference MODEL OUTPUT.safetensors PROMPT_IDS DECODE_IDS [cpu|gpu] [--single-row]");
        }
        const std::string device = argc >= 6 ? argv[5] : "cpu";
        if (device != "cpu" && device != "gpu") throw std::runtime_error("device must be cpu or gpu");
        auto prompt = tokens(argv[3]);
        const auto decode = tokens(argv[4]);
        if (prompt.empty()) throw std::runtime_error("prompt must not be empty");
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = device == "gpu" ? -1 : 0;
        using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        Model model(llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("llama.cpp could not load the artifact");
        const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (const auto & input : {prompt, decode}) {
            for (const auto token : input) {
                if (token >= vocabulary) throw std::runtime_error("token outside vocabulary");
            }
        }
        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(prompt.size() + decode.size() + 8);
        context_params.n_batch = static_cast<uint32_t>(prompt.size());
        context_params.n_ubatch = argc == 7 ? 1 : context_params.n_batch;
        context_params.n_threads = 8;
        context_params.n_threads_batch = 8;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
        Context context(llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("llama.cpp could not create a context");
        std::vector<float> rows;
        const auto append = [&](std::vector<llama_token> & input) {
            auto batch = llama_batch_get_one(input.data(), static_cast<int32_t>(input.size()));
            if (llama_decode(context.get(), batch) != 0) {
                throw std::runtime_error("llama.cpp decode failed");
            }
            const float * row = llama_get_logits_ith(context.get(), -1);
            if (!row) throw std::runtime_error("llama.cpp did not return logits");
            rows.insert(rows.end(), row, row + vocabulary);
        };
        append(prompt);
        for (const auto token : decode) {
            std::vector<llama_token> input{token};
            append(input);
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
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        write_u64(output, header.size());
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) {
            throw std::runtime_error("reference exporter requires a little-endian host");
        }
        output.write(reinterpret_cast<const char *>(rows.data()), static_cast<std::streamsize>(total_bytes));
        output.close();
        auto report_path = std::filesystem::path(argv[2]);
        report_path.replace_extension(".json");
        std::ofstream report(report_path);
        report.exceptions(std::ios::failbit | std::ios::badbit);
        report << "{\"input\":{\"token_ids\":";
        write_tokens(report, prompt);
        report << "},\"output\":{\"fed_token_ids\":";
        write_tokens(report, decode);
        report << ",\"tensor_file\":" << json_string(std::filesystem::path(argv[2]).filename().string())
               << "},\"reference\":{\"implementation\":\"llama.cpp\",\"device\":"
               << json_string(device) << ",\"kv_dtype\":\"float32\",\"flash_attention\":false}}\n";
        report.close();
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
