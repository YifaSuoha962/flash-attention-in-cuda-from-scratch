"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
// Pres
// 在 kernel 内部，CUDA 运行时自动注入了这些只读变量，你不用声明直接可用：

// 变量	        类型	 含义	
// blockIdx	    uint3	当前 block 在 grid 中的编号	
// blockDim	    uint3	每个 block 里的线程数（启动时配置）	
// threadIdx	uint3	当前线程在 block 内的编号	
// gridDim	    uint3	grid 里的 block 总数（启动时配置）	

// 每个变量又包含三个属性：.x、.y、.z，它们是 CUDA 提供的三维线程坐标的三个轴，不代表物理空间坐标，具体的语义——"哪个轴代表行、哪个代表列"完全由你在 kernel 里约定。

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

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        a[i] = a[i] * scalar;
    }
}

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

