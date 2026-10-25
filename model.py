"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    // 1. 计算当前线程的全局一维下标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 2. 越界保护：只处理 [0, n) 范围内的元素
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

// Host 端启动方式
// void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n) {
//     if (n <= 0) return;

//     const int threads = 256;
//     const int blocks  = (n + threads - 1) / threads;  // 向上取整

//     vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
// }

# Step 2 - scale_array (not yet solved)
# TODO: implement

# Step 3 - elementwise_exp (not yet solved)
# TODO: implement

# Step 4 - row_max (not yet solved)
# TODO: implement

# Step 5 - row_sum (not yet solved)
# TODO: implement

# Step 6 - dot_product (not yet solved)
# TODO: implement

# Step 7 - matmul (not yet solved)
# TODO: implement

# Step 8 - transpose (not yet solved)
# TODO: implement

# Step 9 - qk_scores (not yet solved)
# TODO: implement

# Step 10 - softmax_rows (not yet solved)
# TODO: implement

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

