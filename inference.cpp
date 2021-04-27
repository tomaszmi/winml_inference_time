#include "inference.h"

#include "common.h"

#include <string_view>
#include <utility>
#include <vector>

namespace {

ml::LearningModelDeviceKind ConvertToLMDevKind(Inference::DeviceType dev_type) {
  switch (dev_type) {
  default:
    return ml::LearningModelDeviceKind::Default;
  case Inference::DeviceType::Default:
    return ml::LearningModelDeviceKind::Default;
  case Inference::DeviceType::Cpu:
    return ml::LearningModelDeviceKind::Cpu;
  case Inference::DeviceType::DirectX:
    return ml::LearningModelDeviceKind::DirectX;
  case Inference::DeviceType::DirectXHighPerformance:
    return ml::LearningModelDeviceKind::DirectXHighPerformance;
  case Inference::DeviceType::DirectXMinPower:
    return ml::LearningModelDeviceKind::DirectXMinPower;
  }
}

std::vector<std::int64_t>
NormalizeTensorShape(const winrt::Windows::Foundation::Collections::IVectorView<std::int64_t>& shape) {
  std::vector<std::int64_t> normalized;
  normalized.reserve(shape.Size());
  for (std::uint32_t i = 0; i < shape.Size(); i++) {
    auto dim = shape.GetAt(i);
    if (dim <= 0) {
      dim = 1;
    }
    normalized.emplace_back(dim);
  }
  return normalized;
}

std::vector<std::int64_t> GetTensorShape(const ml::ILearningModelFeatureDescriptor& feat_desc) {
  if (feat_desc.Kind() != ml::LearningModelFeatureKind::Tensor) {
    throw std::runtime_error("Non-tensor descriptor is not supported");
  }
  return NormalizeTensorShape(feat_desc.as<ml::TensorFeatureDescriptor>().Shape());
}

std::pair<std::wstring_view, std::vector<std::int64_t>>
ExtractTensorInfo(const ml::ILearningModelFeatureDescriptor& feat_desc) {
  std::pair<std::wstring_view, std::vector<std::int64_t>> result{feat_desc.Name(), GetTensorShape(feat_desc)};
  if (feat_desc.as<ml::TensorFeatureDescriptor>().TensorKind() != ml::TensorKind::Float) {
    throw std::runtime_error("non-float tensor type");
  }
  return result;
}

using VectorViewF32 = winrt::Windows::Foundation::Collections::IVectorView<float>;

} // namespace

float FloatResultView::GetAt(std::uint32_t index) const {
  auto cast = static_cast<const VectorViewF32*>(opaque_);
  return cast->GetAt(index);
}

std::uint32_t FloatResultView::Size() const {
  auto cast = static_cast<const VectorViewF32*>(opaque_);
  return cast->Size();
}

std::uint32_t FloatResultView::GetMany(std::uint32_t start_index, std::vector<float>& items) const {
  auto cast = static_cast<const VectorViewF32*>(opaque_);
  winrt::array_view<float> items_view(items);
  return cast->GetMany(start_index, items_view);
}

std::uint32_t FloatResultView::GetMany(std::uint32_t start_index, float* dest, std::uint32_t dest_size) const {
  auto cast = static_cast<const VectorViewF32*>(opaque_);
  winrt::array_view<float> items_view(dest, std::next(dest, dest_size));
  return cast->GetMany(start_index, items_view);
}

struct Inference::Impl {
  Impl(const std::filesystem::path& model_path, Inference::DeviceType dev_type)
      : model{ml::LearningModel::LoadFromFilePath(model_path.native())}, device{ConvertToLMDevKind(dev_type)} {
    ResetSession();
    if (model.InputFeatures().Size() != 1) {
      throw std::runtime_error("a single input feature is expected");
    }
    if (model.OutputFeatures().Size() != 1) {
      throw std::runtime_error("a single input feature is expected");
    }
    in_tensor = ExtractTensorInfo(model.InputFeatures().GetAt(0));
    out_tensor = ExtractTensorInfo(model.OutputFeatures().GetAt(0));
  }

  void ResetSession() {
    if (intra_op_num_threads == 0) {
      session = {model, device};
    } else {
      auto options = ml::LearningModelSessionOptions();
      auto nativeOptions = options.as<ILearningModelSessionOptionsNative>();
      nativeOptions->SetIntraOpNumThreadsOverride(intra_op_num_threads);
      session = {model, device, options};
    }
  }

  ml::LearningModel model;
  ml::LearningModelDevice device;
  ml::LearningModelSession session{nullptr};

  std::pair<std::wstring_view, std::vector<std::int64_t>> in_tensor;
  std::pair<std::wstring_view, std::vector<std::int64_t>> out_tensor;

  std::uint32_t intra_op_num_threads = 0;
};

Inference::Inference(const std::filesystem::path& model_path, Inference::DeviceType dev_type)
    : guts_{std::make_unique<Impl>(model_path, dev_type)} {}

Inference::~Inference() = default;

void Inference::SetIntraOpNumThreads(std::uint32_t threads_count) {
  guts_->intra_op_num_threads = threads_count;
  guts_->ResetSession();
}

const std::vector<std::int64_t>& Inference::GetInputShape() const { return guts_->in_tensor.second; }

const std::vector<std::int64_t>& Inference::GetOutputShape() const { return guts_->out_tensor.second; }

void Inference::Run(const float* data, std::size_t size, FloatResultViewer result_view) {
  ml::LearningModelBinding binding(guts_->session);

  winrt::array_view<const float> frame_data_view(data, static_cast<std::uint32_t>(size));
  binding.Bind(guts_->in_tensor.first, ml::TensorFloat::CreateFromArray(guts_->in_tensor.second, frame_data_view));
  binding.Bind(guts_->out_tensor.first, ml::TensorFloat::Create(guts_->out_tensor.second));

  auto eval_result = guts_->session.Evaluate(binding, L"");

  if (not eval_result.Succeeded()) {
    // https://docs.microsoft.com/en-us/windows/ai/windows-ml/evaluate-model-inputs#device-removal
    const int eval_error_code = eval_result.ErrorStatus();
    if (eval_error_code == DXGI_ERROR_DEVICE_REMOVED or eval_error_code == DXGI_ERROR_DEVICE_RESET) {
      guts_->ResetSession();
      throw std::runtime_error("WinML evaluation failed with a transient error");
    } else {
      throw std::runtime_error("WinML evaluation failed");
    }
  }

  auto result_tensor = eval_result.Outputs().Lookup(guts_->out_tensor.first).as<ml::TensorFloat>();
  auto result_data_view = result_tensor.GetAsVectorView();
  result_view(FloatResultView{&result_data_view});
}