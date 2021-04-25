#pragma once

#include <iosfwd>
#include <string>
#include <vector>

struct GPUDeviceInfo {
  enum class DeviceType { default_device, high_performance_device, min_power_device };

  DeviceType type = DeviceType::default_device;
  std::wstring description;
  unsigned int vendor_id = 0;
  unsigned int device_id = 0;
  std::size_t dedicated_video_memory = 0;
  std::size_t dedicated_system_memory = 0;
  std::size_t shared_system_memory = 0;
  unsigned int adapter_index = 0;
};

std::wostream& operator<<(std::wostream& out, const GPUDeviceInfo& gpu_device);

std::vector<GPUDeviceInfo> EnumerateGPUDevices();

bool DisplayGPUInventory(std::wostream& out);
