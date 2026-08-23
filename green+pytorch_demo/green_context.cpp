#include "green_context.h"

#include <stdexcept>
#include <iostream>


#define CU_CHECK(call)                                      \
do {                                                        \
    CUresult err = call;                                    \
    if (err != CUDA_SUCCESS) {                              \
        const char* msg = nullptr;                          \
        cuGetErrorString(err, &msg);                        \
        std::cerr                                           \
            << "CUDA error: "                              \
            << msg                                         \
            << " at "                                      \
            << __FILE__                                    \
            << ":"                                         \
            << __LINE__                                    \
            << std::endl;                                  \
        throw std::runtime_error(msg);                      \
    }                                                       \
} while(0)



GreenContextManager::GreenContextManager(
    int dit_sms,
    int device,
    bool ignore_sm_coscheduling
)
:
    device_(device),
    dit_ctx_(nullptr),
    vae_ctx_(nullptr),
    dit_stream_(nullptr),
    vae_stream_(nullptr)
    {

        CU_CHECK(
            cuInit(0)
        );


        init_resources(
            dit_sms,
            ignore_sm_coscheduling
        );


        create_contexts();


        create_streams();

}



GreenContextManager::~GreenContextManager()
{

    if (dit_stream_)
    {
        cuStreamDestroy(
            dit_stream_
        );
    }


    if (vae_stream_)
    {
        cuStreamDestroy(
            vae_stream_
        );
    }


    if (dit_ctx_)
    {
        cuGreenCtxDestroy(
            dit_ctx_
        );
    }


    if (vae_ctx_)
    {
        cuGreenCtxDestroy(
            vae_ctx_
        );
    }

}



void GreenContextManager::init_resources(
    int dit_sms,
    bool ignore_sm_coscheduling
)
{

    CUdevice dev;


    CU_CHECK(
        cuDeviceGet(
            &dev,
            device_
        )
    );


    // Step 1
    CU_CHECK(
        cuDeviceGetDevResource(
            dev,
            &total_resource_,
            CU_DEV_RESOURCE_TYPE_SM
        )
    );


    if (dit_sms <= 0 ||
        static_cast<unsigned int>(dit_sms) >= total_resource_.sm.smCount)
    {
        throw std::invalid_argument(
            "dit_sms must be greater than 0 and smaller than the total SM count"
        );
    }


    // Step 2
    unsigned int groups = 1;


    const unsigned int split_flags = ignore_sm_coscheduling
        ? CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
        : 0U;


    CU_CHECK(
        cuDevSmResourceSplitByCount(
            &dit_resource_,
            &groups,
            &total_resource_,
            &vae_resource_,
            split_flags,
            static_cast<unsigned int>(dit_sms)
        )
    );


    if (groups != 1)
    {
        throw std::runtime_error(
            "cuDevSmResourceSplitByCount did not create exactly one DiT group"
        );
    }


    std::cout
        << "Green Context SM split: requested_dit="
        << dit_sms
        << ", actual_dit="
        << dit_resource_.sm.smCount
        << ", actual_vae="
        << vae_resource_.sm.smCount
        << ", total="
        << total_resource_.sm.smCount
        << ", ignore_sm_coscheduling="
        << (ignore_sm_coscheduling ? "true" : "false")
        << std::endl;


    // Step 3

    CU_CHECK(
        cuDevResourceGenerateDesc(
            &dit_desc_,
            &dit_resource_,
            1
        )
    );


    CU_CHECK(
        cuDevResourceGenerateDesc(
            &vae_desc_,
            &vae_resource_,
            1
        )
    );

}



void GreenContextManager::create_contexts()
{

    CUdevice dev;


    CU_CHECK(
        cuDeviceGet(
            &dev,
            device_
        )
    );



    // ==========================
    // Step 4:
    // create Green Context
    // ==========================

    CU_CHECK(
        cuGreenCtxCreate(
            &dit_ctx_,
            dit_desc_,
            dev,
            CU_GREEN_CTX_DEFAULT_STREAM
        )
    );


    CU_CHECK(
        cuGreenCtxCreate(
            &vae_ctx_,
            vae_desc_,
            dev,
            CU_GREEN_CTX_DEFAULT_STREAM
        )
    );

}



void GreenContextManager::create_streams()
{


    CU_CHECK(
        cuGreenCtxStreamCreate(
            &dit_stream_,
            dit_ctx_,
            CU_STREAM_NON_BLOCKING,
            0
        )
    );


    CU_CHECK(
        cuGreenCtxStreamCreate(
            &vae_stream_,
            vae_ctx_,
            CU_STREAM_NON_BLOCKING,
            0
        )
    );

}



uintptr_t GreenContextManager::get_dit_stream()
{
    return reinterpret_cast<uintptr_t>(
        dit_stream_
    );
}


uintptr_t GreenContextManager::get_vae_stream()
{
    return reinterpret_cast<uintptr_t>(
        vae_stream_
    );
}