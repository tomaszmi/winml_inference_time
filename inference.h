#pragma once

#include <filesystem>
#include <functional>
#include <memory>

class FloatResultView {
public:
  explicit FloatResultView(const void* opaque) : opaque_{opaque} {}

  FloatResultView(const FloatResultView&) = delete;
  FloatResultView(FloatResultView&&) = delete;
  FloatResultView& operator=(const FloatResultView&) = delete;
  FloatResultView& operator=(FloatResultView&&) = delete;

  float GetAt(std::uint32_t index) const;
  std::uint32_t Size() const;
  std::uint32_t GetMany(std::uint32_t start_index, std::vector<float>& items) const;
  std::uint32_t GetMany(std::uint32_t start_index, float* dest, std::uint32_t dest_size) const;

private:
  const void* opaque_;
};

class Inference {
public:
  enum class DeviceType { Default, Cpu, DirectX, DirectXHighPerformance, DirectXMinPower, Custom };

  explicit Inference(const std::filesystem::path& model_path, DeviceType dev_type = DeviceType::Default);
  ~Inference();

  /**
	Value 0 sets default number of threads.
   */
  void SetIntraOpNumThreads(std::uint32_t threads_count);

  const std::vector<std::int64_t>& GetInputShape() const;
  const std::vector<std::int64_t>& GetOutputShape() const;

  using FloatResultViewer = std::function<void(const FloatResultView&)>;
  void Run(const float* data, std::size_t size, FloatResultViewer result_view);

private:
  struct Impl;
  std::unique_ptr<Impl> guts_;
};