#ifndef SAFEMLX_IOS_H
#define SAFEMLX_IOS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ModelHandle ModelHandle;
typedef void (*safemlx_text_callback)(const uint8_t *bytes, size_t length, void *context);

ModelHandle *safemlx_model_create(
    const char *model_path,
    const char *metallib_path,
    char **error_out
);

int32_t safemlx_model_generate(
    ModelHandle *handle,
    const char *prompt,
    safemlx_text_callback callback,
    void *context,
    uint64_t *generated_tokens_out,
    double *ttft_seconds_out,
    double *tokens_per_second_out,
    char **error_out
);

void safemlx_model_free(ModelHandle *handle);
void safemlx_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
