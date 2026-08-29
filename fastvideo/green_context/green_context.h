#pragma once

#include <cuda.h>

#include <cstdint>

class GreenContextManager {
 public:
  GreenContextManager(int dit_sms, int device = 0,
                      bool ignore_sm_coscheduling = false);
  ~GreenContextManager();

  GreenContextManager(const GreenContextManager&) = delete;
  GreenContextManager& operator=(const GreenContextManager&) = delete;

  std::uintptr_t get_dit_stream() const;
  std::uintptr_t get_vae_stream() const;
  unsigned int get_dit_sm_count() const;
  unsigned int get_vae_sm_count() const;
  unsigned int get_total_sm_count() const;

 private:
  void init_resources(int dit_sms, bool ignore_sm_coscheduling);
  void create_contexts();
  void create_streams();

  int device_;
  CUdevResource total_resource_{};
  CUdevResource dit_resource_{};
  CUdevResource vae_resource_{};
  CUdevResourceDesc dit_desc_{};
  CUdevResourceDesc vae_desc_{};
  CUgreenCtx dit_ctx_{};
  CUgreenCtx vae_ctx_{};
  CUstream dit_stream_{};
  CUstream vae_stream_{};
};
