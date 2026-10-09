uint index = thread_position_in_grid.x;
if (index >= ELEMENTS) return;
uint row = index / WIDTH;
uint channel = index % WIDTH;
float sum = 0.0f;
for (int depth = 0; depth < DEPTH; ++depth) {
    int code = codes[row * DEPTH + depth];
    if (code < 0) code += VOCABULARY;
    // The enclosing submission validates the entire index domain. Invalid
    // candidates remain memory-safe until that completion rejects publication.
    float value = 0.0f;
    if (code >= 0 && code < VOCABULARY) {
        value = table[(size_t(depth) * VOCABULARY + code) * WIDTH + channel];
    }
    sum = sum + value;
}
output[index] = sum;
