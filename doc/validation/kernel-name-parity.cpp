// Reference helper: Copyright © 2023-2024 Apple Inc.
// Exact MLX v0.32.0 metal/utils.h helper, retained here solely as an independent oracle.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
namespace reference {
template <typename T>
constexpr bool is_numeric_except_char = std::is_arithmetic_v<T> &&
    !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
    !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t>;

template <typename T>
void concatenate(std::string& acc, T first) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
}

template <typename T, typename... Args>
void concatenate(std::string& acc, T first, Args... args) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
  concatenate(acc, args...);
}
}
#include "mlx/backend/common/kernel_name.h"
namespace candidate = mlx::core;
std::uint64_t comparisons = 0;
template<class... A> void check(A... args) {
  std::string a="prefix_", b=a;
  reference::concatenate(a,args...); candidate::concatenate(b,args...);
  if(a!=b) { std::fprintf(stderr,"mismatch: %s != %s\n",a.c_str(),b.c_str()); std::abort(); }
  ++comparisons;
}
std::uint64_t random_state=37;
std::uint64_t random_value() {
 random_state^=random_state<<13; random_state^=random_state>>7; random_state^=random_state<<17; return random_state;
}
template<bool Fold> auto bench(std::size_t rounds) {
 std::uint64_t checksum=0;
 auto start=std::chrono::steady_clock::now();
 for(std::size_t i=0;i<rounds;++i) {
  std::string name="steel_attention_";
  auto append=[&](auto... args) {
   if constexpr(Fold) candidate::concatenate(name,args...);
   else reference::concatenate(name,args...);
  };
  append("float32_",std::string("float32"),"_bq",int(i%128),"_bk",64,"_bd",128,"_wm",4,"_wn",2,"_mask",bool(i%2));
  checksum+=name.size()+static_cast<unsigned char>(name.back());
 }
 return std::pair{std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),checksum};
}
int main() {
 check(std::string("affine_qmm_t_nax_"), std::string("float32"), "_gs_", 64, "_b_", 4, "_bm", 64, "_bn", 64, "_bk", 64, "_wm", 2, "_wn", 2, "_alN_", true);
 check(std::string("é🙂"), "_", std::string("Σ漢字"));
 check(std::numeric_limits<int>::min(),std::numeric_limits<int>::max(),std::numeric_limits<unsigned>::max());
 check(std::numeric_limits<long long>::min(),std::numeric_limits<long long>::max(),std::numeric_limits<unsigned long long>::max());
 check(-0.0,0.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN());
 check(char('a'),static_cast<signed char>('b'),static_cast<unsigned char>('c'),wchar_t('d'),true,false);
 for(std::size_t i=0;i<100000;++i) {
  const auto n=random_value();
  std::string base="base_"+std::string(n%200, char('a'+n%26));
  check(base,"_align_",bool(n&1),"_rank_",int(n%128),"_size_",n,"_value_",double(n%90000)/19.0,"_tail");
  std::string a=base,b=base;
  reference::concatenate(a,a,a); candidate::concatenate(b,b,b);
  if(a!=b) std::abort(); ++comparisons;
 }
 std::printf("comparisons=%llu\n",static_cast<unsigned long long>(comparisons));
 for(int i=0;i<5;++i) {
  const auto a=bench<false>(400000),b=bench<true>(400000);
  if(a.second!=b.second)std::abort();
  std::printf("round=%d original_s=%.6f fold_s=%.6f ratio=%.4f checksum=%llu\n",i,a.first,b.first,b.first/a.first,static_cast<unsigned long long>(a.second));
 }
}
