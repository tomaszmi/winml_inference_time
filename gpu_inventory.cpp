#include "gpu_inventory.h"

#include <cassert>
#include <ostream>

#include "common.h"
#include <Windows.Graphics.DirectX.Direct3D11.interop.h>
#include <Windows.Graphics.DirectX.h>

#include <dxgi1_6.h>

// taken from https://github.com/microsoft/DirectX-Headers
#include "d3d12.h"
#include "d3dx12.h"

namespace {

const char* DeviceTypeToString(GPUDeviceInfo::DeviceType gpu_dev_type) {
  switch (gpu_dev_type) {
  default:
    assert(false);
    return "UNKNOWN";
  case GPUDeviceInfo::DeviceType::default_device:
    return "default";
  case GPUDeviceInfo::DeviceType::high_performance_device:
    return "high performance";
  case GPUDeviceInfo::DeviceType::min_power_device:
    return "minimum power";
  }
}

bool IsMicrosoftBasicRenderDriver(const DXGI_ADAPTER_DESC1& desc) {
  // If this driver is enumerated it usually means that there is a problem with the graphic drivers.
  return desc.VendorId == 0x1414 and desc.DeviceId == 0x008c;
}

GPUDeviceInfo::DeviceType ToDeviceType(DXGI_GPU_PREFERENCE dev_preference) {
  switch (dev_preference) {
  default:
    throw std::runtime_error("unexpected DXGI_GPU_PREFERENCE enumeration value");
  case DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED:
    return GPUDeviceInfo::DeviceType::default_device;
  case DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_MINIMUM_POWER:
    return GPUDeviceInfo::DeviceType::min_power_device;
  case DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE:
    return GPUDeviceInfo::DeviceType::high_performance_device;
  }
}

void EnumerateGPUByPreference(winrt::com_ptr<IDXGIFactory6>& factory, DXGI_GPU_PREFERENCE dev_preference,
                              std::vector<GPUDeviceInfo>& devs_info) {
  winrt::com_ptr<IDXGIAdapter1> adapter;
  for (unsigned int adapter_index = 0;
       factory->EnumAdapterByGpuPreference(adapter_index, dev_preference, __uuidof(IDXGIAdapter1),
                                           adapter.put_void()) != DXGI_ERROR_NOT_FOUND;
       adapter_index++) {
    DXGI_ADAPTER_DESC1 desc;
    winrt::check_hresult(adapter->GetDesc1(&desc));
    adapter = {};
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE or IsMicrosoftBasicRenderDriver(desc)) {
      // Don't use the Basic Render Driver adapter.
      continue;
    }
    devs_info.emplace_back(GPUDeviceInfo{ToDeviceType(dev_preference), desc.Description, desc.VendorId, desc.DeviceId,
                                         desc.DedicatedVideoMemory, desc.DedicatedSystemMemory, desc.SharedSystemMemory,
                                         adapter_index});
  }
}

} // namespace

std::wostream& operator<<(std::wostream& out, const GPUDeviceInfo& gpu_device) {
  return out << L"description: " << gpu_device.description << L", type: " << DeviceTypeToString(gpu_device.type)
             << L", vendor_id: " << gpu_device.vendor_id << L", device_id: " << gpu_device.device_id
             << L", dedicated_video_memory: " << gpu_device.dedicated_video_memory << L", dedicated_system_memory: "
             << gpu_device.dedicated_system_memory << L", shared_system_memory: " << gpu_device.shared_system_memory;
}

std::vector<GPUDeviceInfo> EnumerateGPUDevices() {
  winrt::com_ptr<IDXGIFactory6> factory;
  winrt::check_hresult(CreateDXGIFactory(__uuidof(IDXGIFactory6), factory.put_void()));

  std::vector<GPUDeviceInfo> devs_info;
  EnumerateGPUByPreference(factory, DXGI_GPU_PREFERENCE_UNSPECIFIED, devs_info);
  EnumerateGPUByPreference(factory, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, devs_info);
  EnumerateGPUByPreference(factory, DXGI_GPU_PREFERENCE_MINIMUM_POWER, devs_info);
  return devs_info;
}

bool DisplayGPUInventory(std::wostream& out) {
  const auto gpu_devs = EnumerateGPUDevices();
  if (gpu_devs.empty()) {
    out << "No GPU device available" << std::endl;
    return false;
  }
  for (auto& gpu_dev : gpu_devs) {
    out << "GPU device: " << gpu_dev << std::endl;
  }
  return true;
}