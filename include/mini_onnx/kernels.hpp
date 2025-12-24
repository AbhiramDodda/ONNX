#pragma once

#include "graph.hpp"
#include <functional>
#include <unordered_map>
#include <memory>

namespace mini_onnx {

// interface
class Kernel {
public:
    virtual ~Kernel() = default;
    virtual void execute(Node& node) = 0;
    virtual std::string op_type() const = 0;
};

// registry
class KernelRegistry {
public:
    KernelRegistry();
    void register_kernel(const std::string& op_type, 
                        std::shared_ptr<Kernel> kernel);
    std::shared_ptr<Kernel> get_kernel(const std::string& op_type) const;
    bool has_kernel(const std::string& op_type) const;
    
    // Register default kernels
    void register_default_kernels();
    
private:
    std::unordered_map<std::string, std::shared_ptr<Kernel>> kernels_;
};

// Matrix Multiplication
// Y = alpha * A * B + beta * C
class GemmKernel : public Kernel {
public:
    void execute(Node& node) override;
    std::string op_type() const override { return "Gemm"; }
    
private:
    void gemm(const float* A, const float* B, const float* C, float* Y,
             int64_t M, int64_t N, int64_t K,
             float alpha, float beta, bool transA, bool transB);
};
// Relu: Rectified Linear Unit
// Y = max(0, X)
class ReluKernel : public Kernel {
public:
    void execute(Node& node) override;
    std::string op_type() const override { return "Relu"; }
};
// Add: Element-wise addition
// Y = A + B (with broadcasting)
class AddKernel : public Kernel {
public:
    void execute(Node& node) override;
    std::string op_type() const override { return "Add"; }
};
// MatMul
// Y = A @ B
class MatMulKernel : public Kernel {
public:
    void execute(Node& node) override;
    std::string op_type() const override { return "MatMul"; }
private:
    void matmul(const float* A, const float* B, float* Y,
               int64_t M, int64_t N, int64_t K);
};
// Conv: 2D Convolution (basic implementation)
// Y = Conv(X, W, B)
class ConvKernel : public Kernel {
public:
    void execute(Node& node) override;
    std::string op_type() const override { return "Conv"; }
    
private:
    void conv2d(const float* input, const float* weights, const float* bias,
               float* output, const std::vector<int64_t>& input_shape,
               const std::vector<int64_t>& weight_shape,
               const std::vector<int64_t>& output_shape,
               const std::vector<int64_t>& kernel_shape,
               const std::vector<int64_t>& strides,
               const std::vector<int64_t>& pads);
};

} 