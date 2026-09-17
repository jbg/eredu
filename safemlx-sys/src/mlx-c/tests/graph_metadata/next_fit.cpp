// Exercises the physical allocator through its existing public native API.
// No test-only cursor access, unbounded fallback or additional grant is used.
namespace {
constexpr size_t next_fit_payload = 64;
constexpr size_t next_fit_alignment = alignof(std::max_align_t);
constexpr size_t next_fit_slots = 8;

struct NextFitBlocks {
  struct Slot {
    void* pointer = nullptr;
    size_t bytes = 0;
    unsigned char tag = 0;
  };
  submission::GraphQuota* quota;
  std::array<Slot, next_fit_slots> slots{};

  explicit NextFitBlocks(submission::GraphQuota* value) : quota(value) {}
  ~NextFitBlocks() { clear(); }
  void* put(size_t index, size_t bytes, unsigned char tag) {
    REQUIRE(slots[index].pointer == nullptr);
    auto* pointer = quota->try_allocate(bytes, next_fit_alignment);
    REQUIRE(pointer != nullptr);
    slots[index] = {pointer, bytes, tag};
    std::memset(pointer, tag, bytes);
    return pointer;
  }
  void release(size_t index) {
    auto& slot = slots[index];
    if (slot.pointer) {
      quota->deallocate(slot.pointer, slot.bytes, next_fit_alignment);
      slot.pointer = nullptr;
    }
  }
  void clear() {
    for (size_t i = 0; i < slots.size(); ++i) release(i);
  }
  void check() const {
    for (size_t i = 0; i < slots.size(); ++i) {
      const auto& slot = slots[i];
      if (!slot.pointer) continue;
      bool unchanged = true;
      const auto* bytes = static_cast<const unsigned char*>(slot.pointer);
      for (size_t j = 0; j < slot.bytes; ++j) unchanged &= bytes[j] == slot.tag;
      CHECK(unchanged);
      for (size_t j = i + 1; j < slots.size(); ++j)
        if (slots[j].pointer) CHECK(slot.pointer != slots[j].pointer);
    }
  }
};

size_t next_fit_extent() {
  size_t extent = 0;
  REQUIRE(submission::GraphQuota::minimum_allocation_extent(
      next_fit_payload, next_fit_alignment, extent));
  return extent;
}
void fill_next_fit(NextFitBlocks& blocks, size_t capacity) {
  for (size_t i = 0; i < next_fit_slots; ++i)
    blocks.put(i, next_fit_payload, static_cast<unsigned char>(i + 1));
  CHECK(blocks.quota->occupied_bytes() == capacity);
  CHECK(blocks.quota->try_allocate(1, 1) == nullptr);
  blocks.check();
}
void finish_next_fit(Arena& arena, NextFitBlocks& blocks, Counts& counts) {
  blocks.clear();
  CHECK(arena.value->occupied_bytes() == 0);
  // The final physical block owns its native reference independently of the
  // caller's arena handle, including after all earlier blocks were coalesced.
  blocks.put(0, next_fit_payload, 0xe7);
  arena.reset();
  CHECK(counts.retired.load() == 0);
  blocks.check();
  blocks.clear();
  CHECK(counts.retired.load() == 1);
}
} // namespace

TEST_CASE("graph quota next-fit wraps without losing live blocks or failed-search state") {
  const size_t extent = next_fit_extent();
  const size_t capacity = extent * next_fit_slots;
  Counts counts;
  Arena arena(counts, capacity);
  NextFitBlocks blocks(arena.value);
  fill_next_fit(blocks, capacity);
  const auto first = blocks.slots[0].pointer;
  const auto near = blocks.slots[1].pointer;
  const auto far = blocks.slots[6].pointer;
  blocks.release(1);
  blocks.release(6);
  CHECK(blocks.put(1, next_fit_payload, 0x91) == near);
  // The next search starts at block 2. Freeing block 0 adds a usable hole
  // behind that cursor; the later hole at 6 must be selected before wrapping.
  blocks.release(0);
  const size_t occupied = arena.value->occupied_bytes();
  CHECK(arena.value->try_allocate(std::numeric_limits<size_t>::max(),
                                 next_fit_alignment) == nullptr);
  CHECK(arena.value->occupied_bytes() == occupied);
  blocks.check();
  CHECK(blocks.put(6, next_fit_payload, 0x96) == far);
  CHECK(blocks.put(0, next_fit_payload, 0x90) == first);
  CHECK(arena.value->occupied_bytes() == capacity);
  for (unsigned attempt = 0; attempt < 3; ++attempt) {
    CHECK(arena.value->try_allocate(1, 1) == nullptr);
    CHECK(arena.value->occupied_bytes() == capacity);
    blocks.check();
  }
  // Exhaustion does not poison the allocator or lose a newly freed hole.
  const auto recovered = blocks.slots[4].pointer;
  blocks.release(4);
  CHECK(blocks.put(4, next_fit_payload, 0x94) == recovered);
  blocks.check();
  finish_next_fit(arena, blocks, counts);
}

TEST_CASE("graph quota next-fit repairs cursor through every coalescing direction") {
  const size_t extent = next_fit_extent();
  const size_t capacity = extent * next_fit_slots;
  for (unsigned direction = 0; direction < 3; ++direction) {
    INFO("coalescing direction " << direction);
    Counts counts;
    Arena arena(counts, capacity);
    NextFitBlocks blocks(arena.value);
    fill_next_fit(blocks, capacity);
    const size_t first = direction == 0 ? 3 : 2;
    const size_t combined = direction == 2 ? 3 : 2;
    const auto expected = blocks.slots[first].pointer;
    // Replacing block 3 positions the cursor at 4; replacing block 2
    // positions it at 3. Those headers are deliberately absorbed below.
    const size_t replacement = direction == 1 ? 2 : 3;
    blocks.release(replacement);
    blocks.put(replacement, next_fit_payload, 0xa0);
    if (direction == 0) {
      blocks.release(4);
      blocks.release(3); // next-free header 4 is absorbed by 3
    } else if (direction == 1) {
      blocks.release(2);
      blocks.release(3); // current header 3 is absorbed by previous 2
    } else {
      blocks.release(2);
      blocks.release(4);
      blocks.release(3); // cursor 4 -> surviving 3 -> surviving 2
    }
    CHECK(arena.value->occupied_bytes() == capacity - combined * extent);
    blocks.check();
    CHECK(arena.value->try_allocate(capacity, next_fit_alignment) == nullptr);
    // The same aligned payload offset makes this request use the complete
    // merged extent. A stale interior header cannot satisfy this allocation.
    const size_t bytes = next_fit_payload + (combined - 1) * extent;
    size_t merged = 0;
    REQUIRE(submission::GraphQuota::minimum_allocation_extent(
        bytes, next_fit_alignment, merged));
    REQUIRE(merged == combined * extent);
    CHECK(blocks.put(first, bytes, 0xb0) == expected);
    CHECK(arena.value->occupied_bytes() == capacity);
    blocks.check();
    finish_next_fit(arena, blocks, counts);
  }
}
