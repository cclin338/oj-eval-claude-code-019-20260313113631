#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Optimization: Build K_all and V_all in HBM first to reduce SRAM pressure
    Matrix* K_all = nullptr;
    Matrix* V_all = nullptr;

    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        K_all = matrix_memory_allocator.Allocate("K_all_" + std::to_string(i));
        V_all = matrix_memory_allocator.Allocate("V_all_" + std::to_string(i));
        gpu_sim.Copy(keys[j], K_all, Position::kInGpuHbm);
        gpu_sim.Copy(values[j], V_all, Position::kInGpuHbm);
      } else {
        Matrix* K_temp = matrix_memory_allocator.Allocate("K_temp_" + std::to_string(i) + "_" + std::to_string(j));
        Matrix* V_temp = matrix_memory_allocator.Allocate("V_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(K_all, keys[j], K_temp, 0, Position::kInGpuHbm);
        gpu_sim.Concat(V_all, values[j], V_temp, 0, Position::kInGpuHbm);
        K_all = K_temp;
        V_all = V_temp;
      }
    }

    // Move to SRAM only when needed for computation
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Transpose K_all to get K_all^T: [i+1, d] -> [d, i+1]
    gpu_sim.Transpose(K_all, Position::kInSharedMemory);

    // Compute Q @ K_all^T: [i+1, d] @ [d, i+1] = [i+1, i+1]
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_all, QK);

    // Apply exp for softmax
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp_" + std::to_string(i));
    gpu_sim.MatExp(QK, QK_exp);

    // Compute row-wise softmax
    Matrix* softmax_QK = nullptr;

    for (size_t row = 0; row <= i; ++row) {
      Matrix* row_data = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.GetRow(QK_exp, row, row_data, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.Sum(row_data, row_sum);

      Matrix* row_normalized = matrix_memory_allocator.Allocate("row_norm_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatDiv(row_data, row_sum, row_normalized);

      if (row == 0) {
        softmax_QK = row_normalized;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("softmax_build_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(softmax_QK, row_normalized, temp, 0, Position::kInSharedMemory);
        softmax_QK = temp;
      }
    }

    // Now compute softmax_QK @ V_all: [i+1, i+1] @ [i+1, d] = [i+1, d]
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_QK, V_all, result);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu