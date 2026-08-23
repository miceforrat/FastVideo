#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "green_context.h"


namespace py = pybind11;



class GreenContextWrapper {

public:

    GreenContextWrapper(
        int dit_sms,
        int device,
        bool ignore_sm_coscheduling
    )
    {
        ctx_ = std::make_shared<GreenContextManager>(
            dit_sms,
            device,
            ignore_sm_coscheduling
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
        py::init<int,int,bool>(),
        py::arg("dit_sms"),
        py::arg("device")=0,
        py::arg("ignore_sm_coscheduling")=false
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