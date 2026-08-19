#ifndef EREDU_IOS_H
#define EREDU_IOS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ModelHandle ModelHandle;
typedef void (*eredu_text_callback)(const uint8_t *bytes, size_t length, void *context);

ModelHandle *eredu_model_create(
    const char *model_path,
    const char *metallib_path,
    char **error_out
);

int32_t eredu_model_generate(
    ModelHandle *handle,
    const char *prompt,
    eredu_text_callback callback,
    void *context,
    uint64_t *generated_tokens_out,
    double *ttft_seconds_out,
    double *tokens_per_second_out,
    char **error_out
);

void eredu_model_free(ModelHandle *handle);
void eredu_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
