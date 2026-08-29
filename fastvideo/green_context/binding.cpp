#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <memory>

#include "green_context.h"

namespace py = pybind11;

class GreenContextWrapper {
 public:
  GreenContextWrapper(int dit_sms, int device, bool ignore_sm_coscheduling)
      : context_(std::make_shared<GreenContextManager>(
            dit_sms, device, ignore_sm_coscheduling)) {}

  std::uintptr_t dit_stream() const { return context_->get_dit_stream(); }
  std::uintptr_t vae_stream() const { return context_->get_vae_stream(); }
  unsigned int dit_sm_count() const {
    return context_->get_dit_sm_count();
  }
  unsigned int vae_sm_count() const {
    return context_->get_vae_sm_count();
  }
  unsigned int total_sm_count() const {
    return context_->get_total_sm_count();
  }

 private:
  std::shared_ptr<GreenContextManager> context_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<GreenContextWrapper>(module, "GreenContext")
      .def(py::init<int, int, bool>(), py::arg("dit_sms"),
           py::arg("device") = 0,
           py::arg("ignore_sm_coscheduling") = false)
      .def("dit_stream", &GreenContextWrapper::dit_stream)
      .def("vae_stream", &GreenContextWrapper::vae_stream)
      .def("dit_sm_count", &GreenContextWrapper::dit_sm_count)
      .def("vae_sm_count", &GreenContextWrapper::vae_sm_count)
      .def("total_sm_count", &GreenContextWrapper::total_sm_count);
}
