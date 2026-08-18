# Vendored native sources

`mlx-v0.32.0.tar.gz` contains MLX v0.32.0 at commit
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`. Its upstream URL and SHA-256 are:

```text
https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.0.tar.gz
8e74a0eb613861ce50c09402dd99dbc42b65a762e7fdd291caa39f611db978ec
```

CMake verifies the digest before extracting the archive and applies patches
from `src/mlx-c/patches` to the build-tree copy.

Common MLX build dependencies are vendored from the upstream release inputs:

| File | Upstream URL | SHA-256 |
| --- | --- | --- |
| `json-v3.11.3.tar.xz` | `https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz` | `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d` |
| `fmt-12.1.0.tar.gz` | `https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz` | `ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea` |
| `metal-cpp-26.zip` | `https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip` | `4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4` |

Each archive retains its upstream license and attribution files.
