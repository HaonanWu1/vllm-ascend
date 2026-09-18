// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define __aicore__
using inType = uint16_t;
using outType = uint16_t;
constexpr uint32_t SHORT_SEQUENCE_CAPACITY = 8;
constexpr uint32_t FULL_STATE_TILE_DIM = 128;
constexpr uint32_t BF16_NUM_PER_BLOCK = 16;
constexpr uint32_t UB_RESERVE_BYTES = 128;
constexpr uint32_t REPEAT_BYTES = 256;
constexpr uint32_t STATE_BANK_PADDING_BYTES = 128;
template <class T, class U>
auto Ceil(T a, U b) {
  return (a + b - 1) / b;
}
uint32_t GetBlockNum() { return 8; }

struct Lengths {
  std::vector<int32_t> values;
  int32_t GetValue(uint64_t index) { return values.at(index); }
};

struct Plan {
  Lengths cuSeqlensGm_;
  uint32_t B_, NV_ = 16, avgload = 0;
  uint32_t realK_ = 128, realV_ = 128, alignK_ = 128, alignV_ = 128;
  uint32_t vStep_ = 64, ubSize_ = 262144, restUbSize_ = 198528, sequenceCapacity_ = 16;
  bool hasGama_ = true, hasGamaK_ = false, canUseShortSequence_ = false;
  explicit Plan(std::vector<int32_t> lengths) : cuSeqlensGm_{lengths}, B_(lengths.size()) {}

  // INSERT_ACTUAL_KERNEL_METHODS

  void Run() {
    ComputeAvgload();
    SelectShortSequenceBuffers();
  }
  void Original() const { assert(vStep_ == 64 && sequenceCapacity_ == 16 && restUbSize_ == 198528); }
};

int main() {
  int checks = 0;
  for (int length = 1; length <= 8; ++length) {
    for (int batch : {1, 2, 10}) {
      Plan p(std::vector<int32_t>(batch, length));
      p.Run();
      assert(p.vStep_ == 128 && p.sequenceCapacity_ == 8 && p.restUbSize_ == 189312);
      assert(p.avgload == static_cast<uint32_t>((batch * length * 16 + 7) / 8));
      ++checks;
    }
  }
  // CPU-only guard tests: not K15 hardware or model performance tests.
  for (const auto& lengths : std::vector<std::vector<int32_t>>{{0, 0}, {1, 9, 0}, {16, 1, 0}, {-1, 8}}) {
    Plan p(lengths);
    p.Run();
    p.Original();
    ++checks;
  }
  Plan mixed({8, 0, 1, 3});
  mixed.Run();
  assert(mixed.vStep_ == 128);
  ++checks;
  Plan narrow({8});
  narrow.realK_ = narrow.alignK_ = 64;
  narrow.Run();
  narrow.Original();
  ++checks;
  Plan shortV({8});
  shortV.realV_ = shortV.alignV_ = 64;
  shortV.Run();
  shortV.Original();
  ++checks;
  Plan bigHeads({8});
  bigHeads.NV_ = 256;
  bigHeads.Run();
  bigHeads.Original();
  ++checks;
  Plan alreadyFull({8});
  alreadyFull.vStep_ = 128;
  alreadyFull.Run();
  assert(alreadyFull.sequenceCapacity_ == 16);
  ++checks;
  Plan insufficient({8});
  insufficient.ubSize_ = 250879;
  insufficient.Run();
  insufficient.Original();
  ++checks;
  Plan exact({8});
  exact.ubSize_ = 250880;
  exact.Run();
  assert(exact.vStep_ == 128);
  ++checks;
  Plan optional({8});
  optional.hasGamaK_ = true;
  optional.Run();
  assert(optional.vStep_ == 128);
  ++checks;
  Plan noGamma({8});
  noGamma.hasGama_ = false;
  noGamma.Run();
  assert(noGamma.vStep_ == 128);
  ++checks;
  std::cout << "Exact kernel planning methods: " << checks << " CPU checks passed\n";
}
