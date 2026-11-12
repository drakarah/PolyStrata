// src/poly/lend_arena_test.cpp - What lend_begin() relies on, without a card: the buffers laid out one behind the
// other in a region (counted first, laid out second, the same bytes both times), and the pointer members whose
// values are saved and put back, with a buffer that does not grow with the batch kept from the base set.
//
//   g++ -std=c++20 src/poly/lend_arena_test.cpp -o lend_arena_test && ./lend_arena_test
#include "lend_arena.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

using strata::poly::LendArena;
using strata::poly::PtrSet;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

template <class F>
bool throws_length_error(F&& f) {
    try {
        f();
    } catch (const std::length_error&) {
        return true;
    }
    return false;
}

void test_alignment_and_order() {
    LendArena a;
    a.lay_out(0x10000, 1 << 20);
    void* p0 = a.alloc(1);
    void* p1 = a.alloc(256);
    void* p2 = a.alloc(257);
    void* p3 = a.alloc(0);
    void* p4 = a.alloc(4096);
    check(p0 == (void*) 0x10000, "the first buffer is at the region's start");
    check(p1 == (void*) (0x10000 + 256), "one byte takes 256");
    check(p2 == (void*) (0x10000 + 512), "256 bytes take 256");
    check(p3 == (void*) (0x10000 + 1024), "257 bytes take 512");
    check(p4 == (void*) (0x10000 + 1280), "no bytes still take 256, so that two buffers never share an address");
    check(a.used() == 1280 + 4096, "used() is the end of the last buffer");
    check(!a.counting(), "laid out, not counting");
}

void test_count_matches_lay_out() {
    // the pass that counts and the pass that lays out ask for the same buffers: random layouts of up to 40 buffers
    std::mt19937 rng(7);
    for (int round = 0; round < 10000; ++round) {
        std::vector<size_t> sizes((size_t) (rng() % 40));
        for (size_t& s : sizes) s = rng() % 3 == 0 ? rng() % 8 : rng() % (size_t(1) << (rng() % 28));
        LendArena counted;
        counted.count();
        check(counted.counting(), "count() starts counting");
        for (size_t s : sizes) check(counted.alloc(s) != nullptr, "a counted buffer has an address that is not null");
        const size_t need = counted.used();
        check(need % LendArena::kAlign == 0, "the bytes counted are a multiple of the alignment");
        LendArena laid;
        const uintptr_t base = 0x7000'0000'0000ull + (rng() % 512) * LendArena::kAlign;
        laid.lay_out(base, need);   // exactly what was counted
        uintptr_t end = base;
        for (size_t s : sizes) {
            const uintptr_t at = (uintptr_t) laid.alloc(s);
            check(at == end, "a buffer starts where the one before ended");
            check((at - base) % LendArena::kAlign == 0, "a buffer is aligned to the region's start");
            end = at + (s ? (s + LendArena::kAlign - 1) / LendArena::kAlign * LendArena::kAlign : LendArena::kAlign);
        }
        check(end == base + need, "the buffers fill exactly the bytes counted");
        check(laid.used() == need, "used() agrees after the second pass");
    }
}

void test_overflow() {
    LendArena a;
    a.lay_out(0x1000, 512);
    check(a.alloc(256) != nullptr, "the first fits");
    check(a.alloc(256) != nullptr, "the second fits exactly");
    check(throws_length_error([&] { a.alloc(1); }), "a buffer past the end is refused");
    check(a.used() == 512, "a refused buffer takes nothing");
    LendArena b;
    b.lay_out(0x1000, 0);
    check(throws_length_error([&] { b.alloc(0); }), "an empty region has room for nothing, not even no bytes");
    LendArena c;
    c.lay_out(0x1000, 300);
    check(throws_length_error([&] { c.alloc(301); }), "a buffer that rounds up past the end is refused");
    LendArena d;
    d.lay_out(0x1000, 256);
    check(throws_length_error([&] { d.alloc(~size_t(0) - 8); }), "a size that overflows the rounding is refused, not wrapped");
}

void test_lay_out_again() {
    LendArena a;
    a.count();
    a.alloc(1000);
    check(a.used() == 1024, "counted 1000 bytes as 1024");
    a.lay_out(0x2000, 4096);
    check(a.used() == 0, "lay_out() starts again at the region's start");
    check(a.alloc(1000) == (void*) 0x2000, "and hands out its first address");
    a.count();
    check(a.used() == 0, "count() starts again at nothing");
}

void test_ptr_set() {
    float* score = (float*) 0x100;
    int* ids = (int*) 0x200;
    void* other = (void*) 0x300;
    float* unset = nullptr;
    PtrSet set;
    set.add(&score);
    set.add(&ids);
    set.add(&other);
    set.add(&unset);
    check(set.size() == 4, "the set holds what was added, whatever the pointer's type");
    set.save();
    score = (float*) 0x900;
    ids = (int*) 0x901;
    other = nullptr;
    unset = (float*) 0x902;
    check(set.saved((void**) &score) == (void*) 0x100, "saved() is the value at save(), not the member's now");
    check(set.saved((void**) &unset) == nullptr, "a member that held null saved null");
    float* stranger = (float*) 0x777;
    check(set.saved((void**) &stranger) == nullptr, "a member that is not in the set has no saved value");
    set.restore();
    check(score == (float*) 0x100 && ids == (int*) 0x200 && other == (void*) 0x300, "restore() puts the values back");
    check(unset == nullptr, "and null where it was null");
    // save() again takes the new values; clear() forgets the members
    score = (float*) 0xA00;
    set.save();
    score = (float*) 0xB00;
    set.restore();
    check(score == (float*) 0xA00, "a second save() replaces the first");
    set.clear();
    check(set.size() == 0, "clear() empties the set");
    score = (float*) 0xC00;
    set.restore();
    check(score == (float*) 0xC00, "an empty set restores nothing");
    check(set.saved((void**) &score) == nullptr, "and has no saved values");
}

void test_ptr_set_duplicates_and_unsaved() {
    float* a = (float*) 0x10;
    PtrSet set;
    set.add(&a);
    set.restore();   // nothing was saved: nothing is put back, nothing is written
    check(a == (float*) 0x10, "restore() before save() leaves the members as they are");
    check(set.saved((void**) &a) == nullptr, "and saved() has nothing before save()");
    set.add(&a);   // twice in the list
    set.save();
    a = (float*) 0x20;
    set.restore();
    check(a == (float*) 0x10, "a member listed twice is restored to its saved value");
}

// What Engine::fbuf_keep() does, on the same pieces: while the lent set is laid out, a buffer that does not grow with
// the batch is the one it was in the base set and takes no room, and one that does grow takes room in the lent set.
// The counting and the laying-out pass must take the same bytes, or the second would run past its region.
struct Model {
    LendArena arena;
    PtrSet set;
    bool pass = false;
    float *grows = nullptr, *fixed = nullptr, *late = nullptr;
    size_t rows = 0;
    size_t scratch_at = 0;
    float* scratch = nullptr;
    float* fbuf(size_t n) {   // as Engine::fbuf()
        float* p = scratch ? scratch + scratch_at : nullptr;
        scratch_at += (n + 63) & ~size_t(63);
        return p;
    }
    float* keep(float** member, size_t n) {   // as Engine::fbuf_keep()
        if (pass)
            if (void* k = set.saved((void**) member)) return (float*) k;
        return fbuf(n);
    }
    void assign() {
        grows = fbuf(rows * 100);
        fixed = keep(&fixed, 5000);
        late = fbuf(rows * 7);
    }
    size_t lay(bool real, uintptr_t base, size_t capacity) {
        pass = true;
        if (real) arena.lay_out(base, capacity);
        else arena.count();
        scratch = nullptr;
        scratch_at = 0;
        assign();
        scratch = (float*) arena.alloc(scratch_at * sizeof(float));
        scratch_at = 0;
        assign();
        pass = false;
        return arena.used();
    }
};

void test_kept_buffer() {
    Model m;
    // the base set: every buffer is its own
    m.rows = 256;
    m.scratch = nullptr;
    m.assign();
    const size_t base_floats = m.scratch_at;
    std::vector<float> base_store(base_floats + 64);
    m.scratch = base_store.data();
    m.scratch_at = 0;
    m.assign();
    float* base_fixed = m.fixed;
    check(base_fixed != nullptr, "the base set has the fixed buffer");
    m.set.add(&m.grows);
    m.set.add(&m.fixed);
    m.set.add(&m.late);
    m.set.save();
    // the lent set at 8 times the rows: counted, then laid out in exactly what was counted
    m.rows = 2048;
    const size_t need = m.lay(false, 0, 0);
    const size_t with_fixed_in_it = (size_t) 2048 * 100 * 4 + (size_t) 2048 * 7 * 4 + 5000 * 4;
    check(need < with_fixed_in_it, "the fixed buffer takes no room in the lent set");
    check(need >= (size_t) 2048 * 100 * 4 + (size_t) 2048 * 7 * 4, "the ones that grow with the batch do");
    std::vector<char> region(need + 256);
    const uintptr_t base = ((uintptr_t) region.data() + 255) / 256 * 256;
    const size_t laid = m.lay(true, base, need);
    check(laid == need, "the pass that lays out takes the bytes the one that counts did");
    check(m.fixed == base_fixed, "the fixed buffer is the base set's in the lent set");
    check((uintptr_t) m.grows >= base && (uintptr_t) m.late >= base, "the others are in the region");
    check((uintptr_t) m.late + (size_t) 2048 * 7 * 4 <= base + need, "and end inside it");
    check(m.grows != m.late, "and do not share an address");
    // the base set comes back
    m.set.restore();
    check(m.fixed == base_fixed, "restore() leaves the fixed buffer where it was");
    check((uintptr_t) m.grows >= (uintptr_t) base_store.data() && (uintptr_t) m.grows < (uintptr_t) (base_store.data() + base_store.size()),
          "and the others in the base set");
}

void test_kept_buffer_missing_in_base() {
    // a buffer the base set did not make (a null saved value) is made in the lent set
    Model m;
    m.rows = 64;
    m.fixed = nullptr;
    m.set.add(&m.grows);
    m.set.add(&m.fixed);
    m.set.add(&m.late);
    m.set.save();
    m.rows = 128;
    const size_t need = m.lay(false, 0, 0);
    check(need >= (size_t) 128 * 100 * 4 + (size_t) 128 * 7 * 4 + 5000 * 4,
          "a fixed buffer the base set lacks takes room in the lent one");
}

void test_lend_wanted() {
    using strata::poly::lend_wanted;
    check(!lend_wanted(5000, 1150, 256, 0), "no long batch: never");
    check(!lend_wanted(5000, 1150, 256, 256), "a long batch that is not larger: never");
    check(!lend_wanted(5000, 1150, 1024, 512), "a long batch below the small one: never");
    check(lend_wanted(5000, 1150, 256, 2048), "a long read borrows");
    check(!lend_wanted(1149, 1150, 256, 2048), "below the minimum: no");
    check(lend_wanted(1150, 1150, 256, 2048), "at the minimum: yes");
    check(!lend_wanted(200, 100, 256, 2048), "fits one small call: no, whatever the minimum");
    check(!lend_wanted(256, 100, 256, 2048), "exactly one small call: no");
    check(lend_wanted(257, 100, 256, 2048), "one token over a small call: yes");
    check(!lend_wanted(-5, 0, 256, 2048), "a negative count never borrows");
}

void test_long_batch_valid() {
    using strata::poly::long_batch_valid;
    check(!long_batch_valid(0, 256), "off is not valid");
    check(!long_batch_valid(256, 256), "equal to the small batch");
    check(!long_batch_valid(100, 256), "below the small batch");
    check(long_batch_valid(257, 256), "just above");
    check(long_batch_valid(4096, 1024), "the largest prompt batch");
    check(!long_batch_valid(4097, 1024), "above the largest");
    check(!long_batch_valid(16384, 256), "far above");
    check(!long_batch_valid(-1, 256), "negative");
}

}   // namespace

int main() {
    test_lend_wanted();
    test_long_batch_valid();
    test_alignment_and_order();
    test_count_matches_lay_out();
    test_overflow();
    test_lay_out_again();
    test_ptr_set();
    test_ptr_set_duplicates_and_unsaved();
    test_kept_buffer();
    test_kept_buffer_missing_in_base();
    std::printf("lend arena test: %d checks passed\n", checks);
    return 0;
}
