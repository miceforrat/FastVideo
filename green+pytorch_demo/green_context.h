#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


class GreenContextManager {

public:

    GreenContextManager(
        // 实际上是指定dit的资源数量，剩余的资源即为VAE所有
        int dit_sms,
        int device = 0,
        bool ignore_sm_coscheduling = false
    );


    ~GreenContextManager();


    uintptr_t get_dit_stream();

    uintptr_t get_vae_stream();



private:

    void init_resources(
        int dit_sms,
        bool ignore_sm_coscheduling
    );


    void create_contexts();


    void create_streams();



private:

    int device_;


    CUdevResource total_resource_;

    CUdevResource dit_resource_;

    CUdevResource vae_resource_;


    CUdevResourceDesc dit_desc_;

    CUdevResourceDesc vae_desc_;


    CUgreenCtx dit_ctx_;

    CUgreenCtx vae_ctx_;


    CUstream dit_stream_;

    CUstream vae_stream_;

};