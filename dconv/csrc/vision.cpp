// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // dcn-v2
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
}
