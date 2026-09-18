#include "doctest/doctest.h"
#include "mlx/backend/common/kernel_name.h"
#include <limits>
#include <string>
using mlx::core::concatenate;
TEST_CASE("kernel names preserve scalar and character formatting") {
  std::string text = "prefix_";
  concatenate(text, true, '_', -17, '_', 42u, '_', 0.5, '_', std::string("é🙂"));
  CHECK(text == "prefix_1_-17_42_0.500000_é🙂");
  std::string chars;
  concatenate(chars, static_cast<signed char>('a'), static_cast<unsigned char>('b'), wchar_t('c'));
  CHECK(chars == "abc");
}
TEST_CASE("kernel name argument values snapshot a destination alias") {
  std::string text(128, 'x');
  const auto original = text;
  concatenate(text, text, '_', text);
  CHECK(text == original + original + "_" + original);
}
TEST_CASE("kernel names append the complete selected attention argument pack") {
  std::string text;
  concatenate(text, "steel_attention_", std::string("float32"), "_bq", 16,
      "_bk", 64, "_bd", 128, "_wm", 4, "_wn", 2, "_mask", std::string("float32"));
  CHECK(text == "steel_attention_float32_bq16_bk64_bd128_wm4_wn2_maskfloat32");
}
TEST_CASE("kernel names preserve the complete eighteen-argument quantized pack") {
  std::string text;
  concatenate(text, std::string("affine_qmm_t_nax_"), std::string("float32"),
      "_gs_", 64, "_b_", 4, "_bm", 64, "_bn", 64, "_bk", 64,
      "_wm", 2, "_wn", 2, "_alN_", true);
  CHECK(text == "affine_qmm_t_nax_float32_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_1");
}
