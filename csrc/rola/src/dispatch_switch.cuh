#pragma once

// HOST-SIDE COMPILE-TIME DISPATCH, as function templates taking a generic lambda -- see docs/internals/dispatch_switch.md#note-l3

#include <type_traits>
#include <utility>

#include <torch/extension.h>

namespace rola {

//: ONE BOOLEAN AXIS. `body` is called with `std::true_type` or `std::false_type`, so -- see docs/internals/dispatch_switch.md#near-line-14
template <typename Body>
inline void bool_switch(bool value, Body&& body) {
  if (value) {
    body(std::true_type{});
  } else {
    body(std::false_type{});
  }
}

//: ONE INTEGER AXIS OVER A CLOSED SET. The candidate values are template arguments, -- see docs/internals/dispatch_switch.md#near-line-25
template <int... VALUES, typename Body>
inline void int_switch(int value, const char* what, Body&& body) {
  bool matched = false;
  //: An expansion over the candidate set, evaluated left to right: exactly one arm -- see docs/internals/dispatch_switch.md#near-line-31
  (void)std::initializer_list<int>{
      (value == VALUES ? (body(std::integral_constant<int, VALUES>{}), matched = true, 0) : 0)...};
  TORCH_CHECK(matched, what, " is outside the instantiated set, got ", value);
}

}  // namespace rola
