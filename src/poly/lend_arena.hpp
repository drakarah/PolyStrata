// src/poly/lend_arena.hpp - what lend_begin() lays out and puts back (Options::long_prompt_batch): a run of buffers one
// behind the other in a region of VRAM, counted first and laid out second, and a set of pointer members whose values
// are saved and put back.  Plain host code, so that a test can run it without a card.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::poly {

// Buffers of a region, each at a multiple of kAlign bytes from the region's start, in the order they are asked for.
// The first pass only counts (count()): alloc() answers with addresses that are not to be used and used() says how
// many bytes the second pass needs.  The second (lay_out()) hands out the region's addresses and says no, with
// std::length_error, to a buffer that does not fit: the counted pass and the laid-out one ask for the same buffers,
// so that is a bug and not a case.
class LendArena {
public:
    static constexpr size_t kAlign = 256;

    void count() {
        counting_ = true;
        base_ = 0;
        capacity_ = 0;
        at_ = 0;
    }
    void lay_out(uintptr_t base, size_t capacity) {
        counting_ = false;
        base_ = base;
        capacity_ = capacity;
        at_ = 0;
    }
    // A buffer takes at least kAlign bytes, so that two buffers never share an address.
    void* alloc(size_t bytes) {
        if (bytes > ~size_t(0) - kAlign) throw std::length_error("a lent buffer larger than the address space");
        size_t take = (bytes + kAlign - 1) / kAlign * kAlign;
        if (take == 0) take = kAlign;
        const size_t at = at_;
        if (take > ~size_t(0) - at) throw std::length_error("the lent buffers are larger than the address space");
        if (!counting_ && at + take > capacity_) throw std::length_error("the lent buffers do not fit their region");
        at_ += take;
        return counting_ ? (void*) (kAlign + at) : (void*) (base_ + at);
    }
    size_t used() const { return at_; }
    bool counting() const { return counting_; }

private:
    bool counting_ = true;
    uintptr_t base_ = 0;
    size_t capacity_ = 0;
    size_t at_ = 0;
};

// Pointer members (of the engine and of its architecture) that point at one set of buffers and are made to point
// at another for a while.  save() remembers what they hold, restore() puts it back.  saved() is what one of them
// held: a buffer that does not grow with the batch is the saved one in the other set too.
class PtrSet {
public:
    void clear() {
        members_.clear();
        saved_.clear();
    }
    void add(void** member) { members_.push_back(member); }
    template <class T>
    void add(T** member) {
        add(reinterpret_cast<void**>(member));
    }
    void save() {
        saved_.clear();
        for (void** m : members_) saved_.push_back(*m);
    }
    void restore() const {
        for (size_t i = 0; i < members_.size() && i < saved_.size(); ++i) *members_[i] = saved_[i];
    }
    // The value `member` held at save(); null when it is not in the set (or held null).
    void* saved(void** member) const {
        for (size_t i = 0; i < members_.size() && i < saved_.size(); ++i)
            if (members_[i] == member) return saved_[i];
        return nullptr;
    }
    size_t size() const { return members_.size(); }

private:
    std::vector<void**> members_;
    std::vector<void*> saved_;
};

// Whether a read of n tokens borrows: the long batch is larger than the one decode runs with, the read needs more than
// one call of the small batch (else the big set could not make it faster), and it has the minimum of tokens.
inline bool lend_wanted(long long n, long long min_tokens, long long small_rows, long long long_rows) {
    return long_rows > small_rows && n > small_rows && n >= min_tokens;
}

// Whether --long-prompt-batch is taken as given: a batch above the small one, up to the largest prompt batch (4096).
inline bool long_batch_valid(long long long_rows, long long small_rows) {
    return long_rows > small_rows && long_rows <= 4096;
}

}   // namespace strata::poly
