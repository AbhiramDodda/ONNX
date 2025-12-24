#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "mini_onnx/runtime.hpp"
#include "mini_onnx/model.hpp"

using namespace mini_onnx;

TEST_CASE("Tensor operations", "[tensor]") {
    SECTION("Create and allocate tensor") {
        Tensor t("test", {2, 3});
        REQUIRE(t.name() == "test");
        REQUIRE(t.shape().size() == 2);
        REQUIRE(t.shape()[0] == 2);
        REQUIRE(t.shape()[1] == 3);
        REQUIRE(t.size() == 6);
        REQUIRE(!t.is_allocated());
        
        t.allocate();
        REQUIRE(t.is_allocated());
        REQUIRE(t.data() != nullptr);
    }
    
    SECTION("Set and get data") {
        Tensor t("test", {2, 2});
        t.allocate();
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        t.set_data(data);
        auto retrieved = t.get_data();
        REQUIRE(retrieved.size() == 4);
        REQUIRE(retrieved[0] == 1.0f);
        REQUIRE(retrieved[3] == 4.0f);
    }
}

TEST_CASE("Graph construction", "[graph]") {
    Graph graph("test_graph");
    
    SECTION("Add tensors") {
        auto t1 = graph.add_tensor("t1", {2, 3});
        REQUIRE(graph.has_tensor("t1"));
        REQUIRE(graph.get_tensor("t1") == t1);
        
        auto t2 = graph.add_tensor("t2", {3, 4});
        REQUIRE(graph.has_tensor("t2"));
    }
    
    SECTION("Add nodes") {
        auto input = graph.add_tensor("input", {1, 2});
        auto output = graph.add_tensor("output", {1, 2});
        
        auto node = graph.add_node("relu1", "Relu");
        node->add_input(input);
        node->add_output(output);
        
        REQUIRE(node->name() == "relu1");
        REQUIRE(node->op_type() == "Relu");
        REQUIRE(node->inputs().size() == 1);
        REQUIRE(node->outputs().size() == 1);
    }
}

TEST_CASE("Relu kernel", "[kernel]") {
    Runtime runtime;
    ModelDef model;
    model.name = "relu_test";
    
    model.inputs.push_back({"input", {1, 4}});
    model.outputs.push_back({"output", {1, 4}});
    
    NodeDef node;
    node.name = "relu1";
    node.op_type = "Relu";
    node.inputs = {"input"};
    node.outputs = {"output"};
    model.nodes.push_back(node);
    
    runtime.load_model(model);
    std::vector<float> input = {-1.0f, 2.0f, -3.0f, 4.0f};
    runtime.set_input("input", input);
    runtime.run();
    auto output = runtime.get_output("output");
    REQUIRE(output.size() == 4);
    REQUIRE(output[0] == 0.0f);  
    REQUIRE(output[1] == 2.0f); 
    REQUIRE(output[2] == 0.0f);  
    REQUIRE(output[3] == 4.0f);  
}

TEST_CASE("MatMul kernel", "[kernel]") {
    Runtime runtime;
    ModelDef model;
    model.name = "matmul_test";
    model.inputs.push_back({"A", {2, 3}});
    model.initializers.push_back({"B", {3, 2}});
    model.outputs.push_back({"C", {2, 2}});
    model.initializer_data["B"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    NodeDef node;
    node.name = "matmul1";
    node.op_type = "MatMul";
    node.inputs = {"A", "B"};
    node.outputs = {"C"};
    model.nodes.push_back(node);
    runtime.load_model(model);
    std::vector<float> A = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    runtime.set_input("A", A);
    runtime.run();
    
    auto C = runtime.get_output("C");
    REQUIRE(C.size() == 4);
    REQUIRE_THAT(C[0], Catch::Matchers::WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(C[1], Catch::Matchers::WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(C[2], Catch::Matchers::WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(C[3], Catch::Matchers::WithinAbs(4.0f, 0.001f));
}

TEST_CASE("Gemm kernel", "[kernel]") {
    Runtime runtime;
    ModelDef model;
    model.name = "gemm_test";
    
    model.inputs.push_back({"A", {2, 3}});
    model.initializers.push_back({"B", {3, 2}});
    model.initializers.push_back({"C", {1, 2}});
    model.outputs.push_back({"Y", {2, 2}});
    
    model.initializer_data["B"] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    model.initializer_data["C"] = {0.5f, 0.5f};
    
    NodeDef node;
    node.name = "gemm1";
    node.op_type = "Gemm";
    node.inputs = {"A", "B", "C"};
    node.outputs = {"Y"};
    node.attributes["alpha"] = "2.0";
    node.attributes["beta"] = "1.0";
    model.nodes.push_back(node);
    
    runtime.load_model(model); 
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    runtime.set_input("A", A);
    runtime.run();
    auto Y = runtime.get_output("Y");
    REQUIRE(Y.size() == 4);
    // Y = 2.0 * A @ B + 1.0 * C
    // A @ B = [[1, 2], [4, 5]]
    // 2 * A @ B = [[2, 4], [8, 10]]
    // Y = [[2.5, 4.5], [8.5, 10.5]]
    REQUIRE_THAT(Y[0], Catch::Matchers::WithinAbs(2.5f, 0.001f));
    REQUIRE_THAT(Y[1], Catch::Matchers::WithinAbs(4.5f, 0.001f));
    REQUIRE_THAT(Y[2], Catch::Matchers::WithinAbs(8.5f, 0.001f));
    REQUIRE_THAT(Y[3], Catch::Matchers::WithinAbs(10.5f, 0.001f));
}

TEST_CASE("Topological sort", "[graph]") {
    Graph graph("test");
    
    auto input = graph.add_tensor("input", {1, 2});
    auto t1 = graph.add_tensor("t1", {1, 2});
    auto t2 = graph.add_tensor("t2", {1, 2});
    auto output = graph.add_tensor("output", {1, 2});
    
    auto n1 = graph.add_node("n1", "Relu");
    n1->add_input(input);
    n1->add_output(t1);
    
    auto n2 = graph.add_node("n2", "Relu");
    n2->add_input(t1);
    n2->add_output(t2);
    
    auto n3 = graph.add_node("n3", "Relu");
    n3->add_input(t2);
    n3->add_output(output);
    
    auto sorted = graph.topological_sort();
    
    REQUIRE(sorted.size() == 3);
    REQUIRE(sorted[0]->name() == "n1");
    REQUIRE(sorted[1]->name() == "n2");
    REQUIRE(sorted[2]->name() == "n3");
}

TEST_CASE("Model loading from JSON string", "[model]") {
    std::string json_str = R"({
        "name": "test_model",
        "inputs": [{"name": "x", "shape": [1, 2]}],
        "outputs": [{"name": "y", "shape": [1, 2]}],
        "nodes": [{
            "name": "relu",
            "op_type": "Relu",
            "inputs": ["x"],
            "outputs": ["y"]
        }]
    })";
    
    auto model = ModelLoader::load_from_string(json_str);
    REQUIRE(model.name == "test_model");
    REQUIRE(model.inputs.size() == 1);
    REQUIRE(model.outputs.size() == 1);
    REQUIRE(model.nodes.size() == 1);
    REQUIRE(model.nodes[0].op_type == "Relu");
}

TEST_CASE("Memory planning", "[runtime]") {
    Runtime runtime;
    
    ModelDef model;
    model.name = "memory_test";
    model.inputs.push_back({"input", {1, 10}});
    model.outputs.push_back({"output", {1, 10}});
    
    NodeDef n1;
    n1.name = "relu1";
    n1.op_type = "Relu";
    n1.inputs = {"input"};
    n1.outputs = {"t1"};
    model.nodes.push_back(n1);
    NodeDef n2;
    n2.name = "relu2";
    n2.op_type = "Relu";
    n2.inputs = {"t1"};
    n2.outputs = {"output"};
    model.nodes.push_back(n2);
    runtime.load_model(model);
    runtime.plan_memory();
    const auto& plan = runtime.memory_plan();
    REQUIRE(plan.total_memory > 0);
    REQUIRE(plan.tensor_offsets.size() >= 2);
}