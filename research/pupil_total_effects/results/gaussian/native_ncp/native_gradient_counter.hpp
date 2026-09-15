#ifndef PUPIL_GRADIENT_COUNTER_HPP
#define PUPIL_GRADIENT_COUNTER_HPP
#include <stan/math/rev.hpp>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace pupil_counter {
inline std::atomic<std::uint64_t> evaluations{0};
struct final_receipt {
  ~final_receipt() noexcept {
    try {
      if (const char* directory = std::getenv("PUPIL_GRAD_COUNTER_DIR")) {
        std::ofstream out(std::string(directory) + "/process-" +
                          std::to_string(getpid()) + ".tsv");
        out << "pid\tgradient_evaluations\n" << getpid() << '\t'
            << evaluations.load(std::memory_order_relaxed) << '\n';
        if (!out) std::cerr << "Failed to write pupil gradient receipt\n";
      }
    } catch (const std::exception& error) {
      std::cerr << "Pupil gradient receipt failure: " << error.what() << '\n';
    }
  }
};
inline final_receipt receipt;
}

// Reverse-mode calls carry var; generated quantities use double. This zero
// adds neither density nor derivative. The R consumer verifies the receipts.
template <typename T>
inline stan::return_type_t<T> pupil_record_gradient(const T&, std::ostream*) {
  if constexpr (stan::is_var<T>::value)
    pupil_counter::evaluations.fetch_add(1, std::memory_order_relaxed);
  return 0.0;
}
inline double pupil_gradient_count(std::ostream*) {
  return static_cast<double>(pupil_counter::evaluations.load(std::memory_order_relaxed));
}
#endif
