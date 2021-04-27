#include "common.h"
#include "gpu_inventory.h"
#include "inference.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

namespace {

std::int64_t GetSizeFromShape(const std::vector<std::int64_t>& shape) {
  assert(not shape.empty());
  return std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<std::int64_t>());
}

std::vector<float> CreateRandomInput(const std::vector<std::int64_t>& shape) {
  const auto size = GetSizeFromShape(shape);

  std::vector<float> result;
  result.reserve(static_cast<std::size_t>(size));

  std::random_device device;
  std::mt19937 engine{device()};
  std::uniform_int_distribution<int> dist{0, 255};
  std::generate(result.begin(), result.end(), [&] { return (dist(engine) / 255.f) - 0.5f; });
  return result;
}

void RunAndMeasure(Inference& inference, std::size_t number_of_iterations, std::chrono::milliseconds nap_time,
                   std::ostream& durations_out) {
  std::cout << "Running " << number_of_iterations << " inference iterations with " << nap_time.count()
            << " ms sleep in between" << std::endl;
  for (std::size_t i = 0; i < number_of_iterations; i++) {
    const auto input = CreateRandomInput(inference.GetInputShape());
    const auto out_size = GetSizeFromShape(inference.GetOutputShape());
    std::vector<float> out(static_cast<std::size_t>(out_size), 0.f);
    const auto ts1 = std::chrono::steady_clock::now();
    inference.Run(input.data(), input.size(), [&out](const auto& data_view) {
      auto count = data_view.GetMany(0, out);
      if (count != out.size()) {
        throw std::runtime_error("unable to fetch expected amount of elements");
      }
    });
    const auto ts2 = std::chrono::steady_clock::now();
    durations_out << nap_time.count() << " " << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
                  << std::endl;
    std::this_thread::sleep_for(nap_time);
  }
}

} // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::cerr << "Usage:\n" << argv[0] << " /PATH/TO/THE/MODEL.onnx" << std::endl;
    return 1;
  }

  std::cout << "Using model " << argv[1] << std::endl;

  const bool is_gpu_available = DisplayGPUInventory(std::wcout);
  Inference inference{argv[1], is_gpu_available ? Inference::DeviceType::DirectX : Inference::DeviceType::Default};

  inference.SetIntraOpNumThreads(1);

  std::ofstream durations_out("./winml_durations.txt");
  using namespace std::chrono_literals;
  constexpr std::size_t number_of_iterations = 1000;

  RunAndMeasure(inference, number_of_iterations, 0ms, durations_out);
  RunAndMeasure(inference, number_of_iterations, 50ms, durations_out);
  RunAndMeasure(inference, number_of_iterations, 100ms, durations_out);
  RunAndMeasure(inference, number_of_iterations, 150ms, durations_out);
  RunAndMeasure(inference, number_of_iterations, 200ms, durations_out);

  std::cout << "The inference durations have been saved in ./winml_durations.txt." << std::endl;
  std::cout << "Run `python ./winml_durations.txt` to visualize the results." << std::endl;
} catch (winrt::hresult_error& e) {
  std::wcerr << "hresult_error [" << std::hex << e.code() << "]: " << e.message().c_str() << std::endl;
} catch (std::exception& e) {
  std::wcerr << "error: " << e.what() << std::endl;
} catch (...) {
  std::wcerr << "Unknown error" << std::endl;
}