# GGML scalar block references

`scalar-block-rows.txt` retains the independent oracle used by `eredu-gguf`'s
block-row tests: GGML revision `2145525a4081d66ff1a87cf43ef809f95a85ac0c`, seed
`0x4e58`, three blocks per format, with exact F32 output bits. Architecture
checkpoint tests consume this packaged snapshot through
`gguf::SCALAR_BLOCK_ROWS_REFERENCE`; they require no sibling workspace files.
