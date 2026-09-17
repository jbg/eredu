#include "doctest/doctest.h"
#include "mlx/backend/metal/intrusive_index.h"
#include <array>
#include <cstdint>
#include <map>
#include <random>

namespace {
struct Entry : mlx::core::metal::detail::IndexLinks<Entry> {
  unsigned payload{0};
  Entry() : IndexLinks(nullptr) {}
};
using Index = mlx::core::metal::detail::IntrusiveIndex<Entry>;
int validate(const Entry *node, const Entry *parent, size_t &count) {
  if (!node)
    return 0;
  REQUIRE(node->parent == parent);
  if (node->left)
    REQUIRE(std::less<const void *>{}(node->left->key, node->key));
  if (node->right)
    REQUIRE(std::less<const void *>{}(node->key, node->right->key));
  const auto left = validate(node->left, node, count);
  const auto right = validate(node->right, node, count);
  REQUIRE(left - right >= -1);
  REQUIRE(left - right <= 1);
  REQUIRE(node->height == 1 + std::max(left, right));
  ++count;
  return node->height;
}
void compare(const Index &index,
             const std::map<const void *, Entry *> &expected) {
  REQUIRE(index.size() == expected.size());
  size_t count = 0;
  validate(index.root(), nullptr, count);
  REQUIRE(count == expected.size());
  auto *actual = index.first();
  for (const auto &[key, value] : expected) {
    REQUIRE(actual == value);
    REQUIRE(index.find(key) == value);
    actual = Index::next(actual);
  }
  REQUIRE(actual == nullptr);
}
} // namespace

TEST_CASE("persistent encoder index keeps physical entries across rotations "
          "and successor removal") {
  std::array<Entry, 257> entries{};
  std::map<const void *, Entry *> expected;
  Index index;
  for (unsigned i = 0; i != entries.size(); ++i) {
    auto &entry = entries[i];
    entry.key = &entry;
    entry.payload = 17 + i;
    REQUIRE(index.insert(&entry) == nullptr);
    expected.emplace(entry.key, &entry);
    compare(index, expected);
  }
  while (!expected.empty()) {
    auto *old = index.root();
    const auto saved = old->payload;
    REQUIRE(index.detach(old) == old);
    REQUIRE(old->payload == saved);
    REQUIRE((!old->left && !old->right && !old->parent));
    expected.erase(old->key);
    compare(index, expected);
  }
}

TEST_CASE("persistent encoder index duplicate candidates stay unconsumed and "
          "detached") {
  std::array<Entry, 3> entries{};
  for (auto &entry : entries)
    entry.key = &entries[0];
  entries[0].payload = 11;
  entries[1].payload = 23;
  Index index;
  REQUIRE(index.insert(&entries[0]) == nullptr);
  CHECK(index.insert(&entries[1]) == &entries[0]);
  CHECK(index.size() == 1);
  CHECK(entries[0].payload == 11);
  CHECK(entries[1].payload == 23);
  CHECK((!entries[1].parent && !entries[1].left && !entries[1].right));
  CHECK(index.pop_first() == &entries[0]);
  CHECK(index.pop_first() == nullptr);
  CHECK(index.insert(&entries[1]) == nullptr);
  CHECK(index.find(entries[1].key) == &entries[1]);
  CHECK(index.pop_first() == &entries[1]);
}

TEST_CASE("persistent encoder index mixed insertion lookup and erasure match "
          "an ordinary ordered map") {
  std::array<Entry, 193> entries{};
  for (auto &entry : entries)
    entry.key = &entry;
  std::map<const void *, Entry *> expected;
  Index index;
  std::minstd_rand random(197);
  for (unsigned step = 0; step != 4000; ++step) {
    auto *entry = &entries[random() % entries.size()];
    if (random() & 1) {
      auto [_, inserted] = expected.emplace(entry->key, entry);
      CHECK(index.insert(entry) == (inserted ? nullptr : entry));
    } else if (expected.erase(entry->key)) {
      CHECK(index.detach(entry) == entry);
    } else {
      CHECK(index.find(entry->key) == nullptr);
    }
    compare(index, expected);
  }
  while (auto *entry = index.pop_first())
    expected.erase(entry->key);
  CHECK(expected.empty());
}
