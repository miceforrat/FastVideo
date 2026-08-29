#include "green_context.h"

#include <iostream>
#include <stdexcept>

#define CU_CHECK(call)                                                     \
  do {                                                                     \
    CUresult error = (call);                                               \
    if (error != CUDA_SUCCESS) {                                           \
      const char* message = nullptr;                                       \
      cuGetErrorString(error, &message);                                   \
      throw std::runtime_error(message == nullptr ? "CUDA driver error"   \
                                                  : message);              \
    }                                                                      \
  } while (0)

GreenContextManager::GreenContextManager(
    int dit_sms, int device, bool ignore_sm_coscheduling)
    : device_(device) {
  CU_CHECK(cuInit(0));
  init_resources(dit_sms, ignore_sm_coscheduling);
  create_contexts();
  create_streams();
}

GreenContextManager::~GreenContextManager() {
  if (dit_stream_ != nullptr) {
    cuStreamDestroy(dit_stream_);
  }
  if (vae_stream_ != nullptr) {
    cuStreamDestroy(vae_stream_);
  }
  if (dit_ctx_ != nullptr) {
    cuGreenCtxDestroy(dit_ctx_);
  }
  if (vae_ctx_ != nullptr) {
    cuGreenCtxDestroy(vae_ctx_);
  }
}

void GreenContextManager::init_resources(
    int dit_sms, bool ignore_sm_coscheduling) {
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, device_));
  CU_CHECK(cuDeviceGetDevResource(
      device, &total_resource_, CU_DEV_RESOURCE_TYPE_SM));

  if (dit_sms <= 0 ||
      static_cast<unsigned int>(dit_sms) >= total_resource_.sm.smCount) {
    throw std::invalid_argument(
        "dit_sms must be greater than zero and smaller than total SMs");
  }

  unsigned int groups = 1;
  const unsigned int flags = ignore_sm_coscheduling
                                 ? CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
                                 : 0U;
  CU_CHECK(cuDevSmResourceSplitByCount(
      &dit_resource_, &groups, &total_resource_, &vae_resource_, flags,
      static_cast<unsigned int>(dit_sms)));
  if (groups != 1) {
    throw std::runtime_error(
        "cuDevSmResourceSplitByCount did not create exactly one DiT group");
  }

  CU_CHECK(cuDevResourceGenerateDesc(&dit_desc_, &dit_resource_, 1));
  CU_CHECK(cuDevResourceGenerateDesc(&vae_desc_, &vae_resource_, 1));

  std::cout << "Green Context SM split: requested_dit=" << dit_sms
            << ", actual_dit=" << dit_resource_.sm.smCount
            << ", actual_vae=" << vae_resource_.sm.smCount
            << ", total=" << total_resource_.sm.smCount
            << ", ignore_sm_coscheduling="
            << (ignore_sm_coscheduling ? "true" : "false") << std::endl;
}

void GreenContextManager::create_contexts() {
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, device_));
  CU_CHECK(cuGreenCtxCreate(
      &dit_ctx_, dit_desc_, device, CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(cuGreenCtxCreate(
      &vae_ctx_, vae_desc_, device, CU_GREEN_CTX_DEFAULT_STREAM));
}

void GreenContextManager::create_streams() {
  CU_CHECK(cuGreenCtxStreamCreate(
      &dit_stream_, dit_ctx_, CU_STREAM_NON_BLOCKING, 0));
  CU_CHECK(cuGreenCtxStreamCreate(
      &vae_stream_, vae_ctx_, CU_STREAM_NON_BLOCKING, 0));
}

std::uintptr_t GreenContextManager::get_dit_stream() const {
  return reinterpret_cast<std::uintptr_t>(dit_stream_);
}

std::uintptr_t GreenContextManager::get_vae_stream() const {
  return reinterpret_cast<std::uintptr_t>(vae_stream_);
}

unsigned int GreenContextManager::get_dit_sm_count() const {
  return dit_resource_.sm.smCount;
}

unsigned int GreenContextManager::get_vae_sm_count() const {
  return vae_resource_.sm.smCount;
}

unsigned int GreenContextManager::get_total_sm_count() const {
  return total_resource_.sm.smCount;
}
