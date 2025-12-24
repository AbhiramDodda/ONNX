#include "mini_onnx/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mini_onnx {

// Kernel registry
KernelRegistry::KernelRegistry() {
}

void KernelRegistry::register_kernel(const std::string& op_type,
                                    std::shared_ptr<Kernel> kernel) {
    kernels_[op_type] = kernel;
}

std::shared_ptr<Kernel> KernelRegistry::get_kernel(const std::string& op_type) const {
    auto it = kernels_.find(op_type);
    return it != kernels_.end() ? it->second : nullptr;
}

bool KernelRegistry::has_kernel(const std::string& op_type) const {
    return kernels_.find(op_type) != kernels_.end();
}

void KernelRegistry::register_default_kernels() {
    register_kernel("Gemm", std::make_shared<GemmKernel>());
    register_kernel("Relu", std::make_shared<ReluKernel>());
    register_kernel("Add", std::make_shared<AddKernel>());
    register_kernel("MatMul", std::make_shared<MatMulKernel>());
    register_kernel("Conv", std::make_shared<ConvKernel>());
}

// Gemm kernel
void GemmKernel::execute(Node& node) {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("Gemm requires at least 2 inputs");
    }
    if (node.outputs().size() < 1) {
        throw std::runtime_error("Gemm requires at least 1 output");
    }
    
    auto A = node.inputs()[0];
    auto B = node.inputs()[1];
    auto C = node.inputs().size() > 2 ? node.inputs()[2] : nullptr;
    auto Y = node.outputs()[0];
    
    // Parse attributes
    float alpha = 1.0f;
    float beta = 1.0f;
    bool transA = false;
    bool transB = false;
    
    if (node.has_attribute("alpha")) {
        alpha = std::stof(node.get_attribute("alpha"));
    }
    if (node.has_attribute("beta")) {
        beta = std::stof(node.get_attribute("beta"));
    }
    if (node.has_attribute("transA")) {
        transA = std::stoi(node.get_attribute("transA")) != 0;
    }
    if (node.has_attribute("transB")) {
        transB = std::stoi(node.get_attribute("transB")) != 0;
    }
    
    // Infer output shape and allocate if needed
    const auto& shape_a = A->shape();
    const auto& shape_b = B->shape();
    
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::runtime_error("Gemm only supports 2D matrices");
    }
    
    int64_t M = transA ? shape_a[1] : shape_a[0];
    int64_t K = transA ? shape_a[0] : shape_a[1];
    int64_t N = transB ? shape_b[0] : shape_b[1];
    
    // Set output shape if not set
    if (Y->shape().empty() || Y->shape()[0] == 0) {
        *Y = Tensor(Y->name(), {M, N});
    }
    
    if (!Y->is_allocated()) {
        Y->allocate();
    }
    
    gemm(A->data(), B->data(), C ? C->data() : nullptr, Y->data(),
         M, N, K, alpha, beta, transA, transB);
}

void GemmKernel::gemm(const float* A, const float* B, const float* C, float* Y,
                     int64_t M, int64_t N, int64_t K,
                     float alpha, float beta, bool transA, bool transB) {
    if (C && beta != 0.0f) {
        for (int64_t i = 0; i < M * N; ++i) {
            Y[i] = beta * C[i];
        }
    } else {
        for (int64_t i = 0; i < M * N; ++i) {
            Y[i] = 0.0f;
        }
    }
    
    // Compute Y = alpha * A * B + beta * C
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float a_val = transA ? A[k * M + m] : A[m * K + k];
                float b_val = transB ? B[n * K + k] : B[k * N + n];
                sum += a_val * b_val;
            }
            Y[m * N + n] += alpha * sum;
        }
    }
}

// Relu kernel
void ReluKernel::execute(Node& node) {
    if (node.inputs().size() < 1) {
        throw std::runtime_error("Relu requires 1 input");
    }
    if (node.outputs().size() < 1) {
        throw std::runtime_error("Relu requires 1 output");
    }
    
    auto X = node.inputs()[0];
    auto Y = node.outputs()[0];
    if (Y->shape().empty() || Y->shape()[0] == 0) {
        *Y = Tensor(Y->name(), X->shape());
    }
    
    if (!Y->is_allocated()) {
        Y->allocate();
    }
    
    const float* x_data = X->data();
    float* y_data = Y->data();
    int64_t size = X->size();
    
    for (int64_t i = 0; i < size; ++i) {
        y_data[i] = std::max(0.0f, x_data[i]);
    }
}

// Add kernel
void AddKernel::execute(Node& node) {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("Add requires 2 inputs");
    }
    if (node.outputs().size() < 1) {
        throw std::runtime_error("Add requires 1 output");
    }
    
    auto A = node.inputs()[0];
    auto B = node.inputs()[1];
    auto Y = node.outputs()[0];
    
    // Simple broadcasting
    if (Y->shape().empty() || Y->shape()[0] == 0) {
        *Y = Tensor(Y->name(), A->shape());
    }
    
    if (!Y->is_allocated()) {
        Y->allocate();
    }
    
    const float* a_data = A->data();
    const float* b_data = B->data();
    float* y_data = Y->data();
    
    int64_t a_size = A->size();
    int64_t b_size = B->size();
    
    if (a_size == b_size) {
        // Element-wise add
        for (int64_t i = 0; i < a_size; ++i) {
            y_data[i] = a_data[i] + b_data[i];
        }
    } else if (b_size == 1) {
        // Broadcast B (scalar)
        float b_val = b_data[0];
        for (int64_t i = 0; i < a_size; ++i) {
            y_data[i] = a_data[i] + b_val;
        }
    } else {
        throw std::runtime_error("Add: unsupported broadcast pattern");
    }
}

// MatMul kernel
void MatMulKernel::execute(Node& node) {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("MatMul requires 2 inputs");
    }
    if (node.outputs().size() < 1) {
        throw std::runtime_error("MatMul requires 1 output");
    }
    
    auto A = node.inputs()[0];
    auto B = node.inputs()[1];
    auto Y = node.outputs()[0];
    
    const auto& shape_a = A->shape();
    const auto& shape_b = B->shape();
    
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::runtime_error("MatMul only supports 2D matrices");
    }
    
    int64_t M = shape_a[0];
    int64_t K = shape_a[1];
    int64_t N = shape_b[1];
    
    if (shape_b[0] != K) {
        throw std::runtime_error("MatMul: incompatible dimensions");
    }
    
    if (Y->shape().empty() || Y->shape()[0] == 0) {
        *Y = Tensor(Y->name(), {M, N});
    }
    
    if (!Y->is_allocated()) {
        Y->allocate();
    }
    
    matmul(A->data(), B->data(), Y->data(), M, N, K);
}

void MatMulKernel::matmul(const float* A, const float* B, float* Y,
                         int64_t M, int64_t N, int64_t K) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            Y[m * N + n] = sum;
        }
    }
}

// Conv kernel (basic 2D convolution)
void ConvKernel::execute(Node& node) {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("Conv requires at least 2 inputs (input, weights)");
    }
    if (node.outputs().size() < 1) {
        throw std::runtime_error("Conv requires 1 output");
    }
    
    auto X = node.inputs()[0];  // [N, C, H, W]
    auto W = node.inputs()[1];  // [M, C, kH, kW]
    auto B = node.inputs().size() > 2 ? node.inputs()[2] : nullptr;  // [M]
    auto Y = node.outputs()[0];
    
    std::vector<int64_t> kernel_shape = {3, 3}; 
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};  // top, left, bottom, right
    
    if (node.has_attribute("kernel_shape")) {
        // Parse comma-separated values
        auto ks_str = node.get_attribute("kernel_shape");
        // Simplified parsing
    }
    
    const auto& x_shape = X->shape();
    const auto& w_shape = W->shape();
    
    if (x_shape.size() != 4 || w_shape.size() != 4) {
        throw std::runtime_error("Conv only supports 4D tensors (NCHW)");
    }
    
    int64_t N = x_shape[0];
    int64_t C = x_shape[1];
    int64_t H = x_shape[2];
    int64_t W_dim = x_shape[3];
    
    int64_t M = w_shape[0];
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];
    
    int64_t outH = (H + pads[0] + pads[2] - kH) / strides[0] + 1;
    int64_t outW = (W_dim + pads[1] + pads[3] - kW) / strides[1] + 1;
    
    std::vector<int64_t> out_shape = {N, M, outH, outW};
    
    if (Y->shape().empty() || Y->shape()[0] == 0) {
        *Y = Tensor(Y->name(), out_shape);
    }
    
    if (!Y->is_allocated()) {
        Y->allocate();
    }
    
    conv2d(X->data(), W->data(), B ? B->data() : nullptr, Y->data(),
          x_shape, w_shape, out_shape, kernel_shape, strides, pads);
}

void ConvKernel::conv2d(const float* input, const float* weights, const float* bias,
                       float* output, const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& weight_shape,
                       const std::vector<int64_t>& output_shape,
                       const std::vector<int64_t>& kernel_shape,
                       const std::vector<int64_t>& strides,
                       const std::vector<int64_t>& pads) {
    // Simplified 2D convolution (no padding support for brevity)
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    
    int64_t M = weight_shape[0];
    int64_t kH = weight_shape[2];
    int64_t kW = weight_shape[3];
    
    int64_t outH = output_shape[2];
    int64_t outW = output_shape[3];
    
    // Initialize output
    for (int64_t i = 0; i < N * M * outH * outW; ++i) {
        output[i] = 0.0f;
    }
    
    // Convolution
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float sum = bias ? bias[m] : 0.0f;
                    
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t ih = oh * strides[0] + kh;
                                int64_t iw = ow * strides[1] + kw;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int64_t input_idx = n * C * H * W + c * H * W + ih * W + iw;
                                    int64_t weight_idx = m * C * kH * kW + c * kH * kW + kh * kW + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int64_t output_idx = n * M * outH * outW + m * outH * outW + oh * outW + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

} // namespace mini_onnx