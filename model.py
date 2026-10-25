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

// i 不是"一个值"，而是"每个线程各自的一个值"。CUDA 里没有"全线程共享一份、各自独立的值"这种东西——要么是每线程私有的局部变量（int i），
// 要么是所有线程共享的同一块内存（__device__ int g_i，会打架）。

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = expf(a[i]);   // 用 expf，避免隐式 double 提升
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界保护：grid 通常是向上取整，尾部会有多余线程
    if (r >= rows) return;
    // 从整块矩阵的首地址出发，跳到第 r 行的开头，得到"第 r 行首元素"的指针。
    // 注意用 long long 防止 r*cols 溢出
    const float* row = matrix + (long long)r * cols;
    // 线性扫描求最大；用 -INFINITY 而不是 0 作为初值
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) {
        if (row[c] > m) m = row[c];
    }
    // 写回这一行的结果
    out[r] = m;
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    const float* row = matrix + (long long)r * cols;

    float sum = 0.0f;                        // ✅ float + 正确初值
    for (int c = 0; c < cols; ++c) {
        sum += row[c];
    }
    out[r] = sum;                            // ✅ 写回
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // 1. 累加器必须是局部 float，初值 0.0f
    float acc = 0.0f;

    // 2. 单线程串行扫完 n 个元素，累加 a[i]*b[i]
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }

    // 3. 直接返回标量，不需要写 out[...] 也不需要同步
    return acc;
}

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

