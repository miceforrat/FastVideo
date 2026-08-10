#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "green_context.h"


namespace py = pybind11;



class GreenContextWrapper {

public:

    GreenContextWrapper(
        int dit_sms,
        int device
    )
    {
        ctx_ = std::make_shared<GreenContextManager>(
            dit_sms,
            device
        );
    }



    uintptr_t dit_stream()
    {
        return ctx_->get_dit_stream();
    }


    uintptr_t vae_stream()
    {
        return ctx_->get_vae_stream();
    }


private:

    std::shared_ptr<GreenContextManager> ctx_;

};



PYBIND11_MODULE(
    greenctx,
    m
)
{

    py::class_<GreenContextWrapper>(
        m,
        "GreenContext"
    )


    .def(
        py::init<int,int>(),
        py::arg("dit_sms"),
        py::arg("device")=0
    )


    .def(
        "dit_stream",
        &GreenContextWrapper::dit_stream
    )


    .def(
        "vae_stream",
        &GreenContextWrapper::vae_stream
    );

}