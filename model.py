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
    
    // 从整块矩阵的首地址出发 （matrix ~ matrix[0][0]），跳到第 r 行的开头（+ r * cols），得到"第 r 行首元素"的指针。
    // 为什么用 const ？
    //   1. 编译器优化：可以做只读缓存（__ldg 之类）、避免不必要的重排序。
    //   2. 类型安全：matrix 本身参数类型就是 const float*，从 const float* 得到的指针自然也是 const，不加 const 反而要 const_cast。
    
    // 注意用 long long 防止 r*cols 溢出，触发条件：rows * cols > 2^31 - 1。
    // 对 float 来说，就是矩阵 > 8 GB，现在的 A100/H100 一块卡 40GB/80GB，轻轻松松触发。
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
//     // TODO: write out[r] = sum of matrix row r
//     int r = blockIdx.x * blockDim.x + threadIdx.x;
//     if (r >= rows) return;

//     const float* row = matrix + (long long)r * cols;

//     float sum = 0.0f;                        // ✅ float + 正确初值
//     for (int c = 0; c < cols; ++c) {
//         sum += row[c];
//     }
//     out[r] = sum;                            // ✅ 写回

// 进阶：规约写法
    int r = blockIdx.x;                        // ✅ 一个 block 处理一行
    if (r >= rows) return;

    const float* row = matrix + (long long)r * cols;

    // 1. 每线程的列内部分和
    float partial = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        partial += row[c];
    }

    // 2. 写进共享内存
    // 对于用 extern 定义的变量，编译期不知道有多大；运行期才为其分配内存
    extern __shared__ float smem[];            // 大小由 launch 决定：blockDim.x*sizeof(float)
    smem[threadIdx.x] = partial;                // 每个线程写自己的槽
    __syncthreads();                           // ✅ 必须：让所有线程的 partial 可见，缺了这行，线程 0 可能读到其他线程还没写的 0

    // 3. 树形归约（每轮减半，O(log B) 步）
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();                       // ✅ 每轮之间必须同步
    }

    // 4. 线程 0 写回
    if (threadIdx.x == 0) {
        out[r] = smem[0];
    }
}

// 校验实验
// int main() {
//     const int rows = 17, cols = 1000;          // 覆盖非整除、中等列数
//     const int n = rows * cols;
//     std::vector<float> h_mat(n), h_A(rows), h_B(rows), ref(rows);

//     // 用带正负的随机值填，验证 float 累加精度
//     for (int i = 0; i < n; ++i)
//         h_mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f);
//     for (int r = 0; r < rows; ++r) {
//         double s = 0.0;
//         for (int c = 0; c < cols; ++c) s += h_mat[r * cols + c];
//         ref[r] = static_cast<float>(s);
//     }

//     float *d_mat, *d_A, *d_B;
//     CUDA_CHECK(cudaMalloc(&d_mat, n * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_A, rows * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_B, rows * sizeof(float)));
//     CUDA_CHECK(cudaMemcpy(d_mat, h_mat.data(), n * sizeof(float), cudaMemcpyHostToDevice));

//     const int threads = 256;
//     // A：一线程一行
//     row_sum<<<(rows + threads - 1) / threads, threads>>>(d_mat, d_A, rows, cols);
//     // B：一 block 一行
//     row_sum_block<<<rows, threads, threads * sizeof(float)>>>(d_mat, d_B, rows, cols);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, rows * sizeof(float), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, rows * sizeof(float), cudaMemcpyDeviceToHost));

//     float errA = 0.f, errB = 0.f;
//     for (int r = 0; r < rows; ++r) {
//         errA = std::fmax(errA, std::fabs(h_A[r] - ref[r]));
//         errB = std::fmax(errB, std::fabs(h_B[r] - ref[r]));
//     }
//     printf("A(one-thread-per-row): max_err=%e %s\n", errA, errA < 1e-3f ? "PASS" : "FAIL");
//     printf("B(one-block-per-row) : max_err=%e %s\n", errB, errB < 1e-3f ? "PASS" : "FAIL");

//     CUDA_CHECK(cudaFree(d_mat)); CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
//     return 0;
// }

# Step 6 - dot_product
// 限定符比较
//     限定符	  调用/执行位置	    是否 kernel（可 <<<>>> 启动）	返回值	 本步
// __host__	      host (CPU)	    ❌	                        任意	 普通 C++ 函数
// __device__	  device (GPU)	    ❌	                        任意	 ✅ 本步
// __global__	  hdevice (GPU)	    ✅	                        void	 前面的 kernel
// 关键点：

// __global__ 是"从 host 启动、在 GPU 上跑的 kernel"。它没有返回值（void），通过指针写回结果，必须用 <<<blocks, threads>>> 启动。

// __device__ 是"只能在 GPU 上被调用的普通函数"。它有返回值，不能被 host 调用，也不能被 <<<>>> 启动，只能被 __global__ kernel 或另一个 __device__ 函数调用。
// 并行度由调用它的 kernel 决定，不由它自己决定。

__device__ float dot_product(const float* a, const float* b, int n) {

    // 1. 累加器必须是局部 float，初值 0.0f
    float acc = 0.0f;

    // 2. 单线程串行扫完 n 个元素，累加 a[i]*b[i]
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }

    // 3. 直接返回标量，不需要写 out[...] 也不需要同步, 返回值直接落进调用者的寄存器.
    return acc;
}

// 如果用 __global__ 限定，会报error: a __global__ function must have void return type。
// 因为 __global__ 是"异步启动"的语义——host 发起后立刻返回，GPU 端跑完结果只能通过指针写回，没法用返回值同步拿回来。
// 这正是 __device__ 存在的意义：同步、有返回值的 GPU 端普通函数。

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    // 1. 从 2D 网格拿到本线程负责的输出元素坐标 (i, j)
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // 列号，对应 C 的第二维 n
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // 行号，对应 C 的第一维 m

    // 2. 边界保护：m / n 不是 blockDim 倍数时，尾部线程必须退出
    if (i >= m || j >= n) return;

    // 3. 定位 A 的第 i 行、B 的第 j 列
    //    A 行主序 [m x k]：A[i][l] 在 a + i*k + l，遍历指定vector内的特征，步长 1
    //    B 行主序 [k x n]：B[l][j] 在 b + l*n + j，遍历指定vector内的特征，步长 n  ← 这就是经典坑
    const float* A_row = a + (long long)i * k;       // 长度 k
    const float* B_col = b + j;                      // 从第 j 个元素开始，步长 n

    // 4. 沿共享维度 k 做累加（相当于 dot_product(A_row, B_col 跨步取)）
    float acc = 0.0f;
    for (int l = 0; l < k; ++l) {
        acc += A_row[l] * B_col[l * n];              // B[l][j] = b[l*n + j]
    }

    // 5. 写回 C[i][j] = c + i*n + j, c \in R^{n \times n}
    c[(long long)i * n + j] = acc;  
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    // 1. 2D 网格：threadIdx.x → 列 c，threadIdx.y → 行 r（与 matmul 的约定一致）
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // 输入列号，范围 [0, cols)
    int r = blockIdx.y * blockDim.y + threadIdx.y;   // 输入行号，范围 [0, rows)

    // 2. 边界保护：rows/cols 不一定是 blockDim 倍数
    if (r >= rows || c >= cols) return;

    // 3. 读 in[r][c]，写到 out[c][r]
    //    输入行主序 (rows, cols)：in[r*cols + c]
    //    输出行主序 (cols, rows)：out[c*rows + r]  ← 注意 stride 是 rows，不是 cols
    out[(long long)c * rows + r] = in[(long long)r * cols + c];
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;      // k is transposed, each col is a feature

    if (i < seq_len && j < seq_len) {
        // q 取一行，k 取一列
        const float* q_row = q + i * head_dim;
        const float* k_row = k + j * head_dim;
        
        // dot_prod: [seq_len, seq_len]
        scores[i * seq_len + j] =
            dot_product(q_row, k_row, head_dim) / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
#include <math_constants.h>

__global__ void softmax_rows(float* matrix, int rows, int cols) {
    // 一个 block 处理一行。
    int row = blockIdx.x;

    // 线程在 block 内的编号；用于分配列工作和共享内存位置。
    int tid = threadIdx.x;

    if (row >= rows || cols <= 0) {
        return;
    }

    // 使用静态共享内存，而不是 extern __shared__ 动态共享内存。
    // 动态共享内存需要在 kernel 启动时额外传入字节数；
    // 如果测试框架没传，shared 就没有足够空间，访问 max_values / sum_values 会出错。
    // 静态共享内存由编译器分配，因此启动 kernel 时无需额外传共享内存大小。
    // 两个数组分别保存最大值归约和指数和归约的中间结果。
    // 要求 blockDim.x <= 1024。
    __shared__ float max_values[1024];
    __shared__ float sum_values[1024];

    float* row_data = matrix + row * cols;

    // 每个线程处理 tid、tid + blockDim.x、... 列。
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_data[col]);
    }

    max_values[tid] = local_max;
    __syncthreads();

    // 归约得到该行的最大值。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_values[tid] =
                fmaxf(max_values[tid], max_values[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = max_values[0];

    // 减去行最大值后计算指数，并累计局部和。
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = expf(row_data[col] - row_max);
        row_data[col] = value;
        local_sum += value;
    }

    sum_values[tid] = local_sum;
    __syncthreads();

    // 归约得到该行的指数和。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_values[tid] += sum_values[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sum_values[0];

    // 原地归一化。
    for (int col = tid; col < cols; col += blockDim.x) {
        row_data[col] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]
    // 每个线程负责一个输出元素 out[i, d]。
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // 处理 seq_len 或 head_dim 不是 block 尺寸整数倍的情况。
    if (i >= seq_len || d >= head_dim) {
        return;
    }

    float sum = 0.0f;

    // 沿共享的序列维度做点积：
    // P[i, j] 的行优先索引为 i * seq_len + j；
    // V[j, d] 的行优先索引为 j * head_dim + d。
    for (int j = 0; j < seq_len; ++j) {
        sum += p[i * seq_len + j] * v[j * head_dim + d];
    }

    // out 的形状是 (seq_len, head_dim)，按行优先存储。
    out[i * head_dim + d] = sum;
}

# Step 12 - naive_attention
void naive_attention(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    int seq_len,
    int head_dim
) {
    // 临时保存 QK^T / sqrt(head_dim)，之后原地转换为 softmax 概率。
    float* d_scores = nullptr;
    size_t scores_bytes =
        static_cast<size_t>(seq_len) * seq_len * sizeof(float);

    if (cudaMalloc(reinterpret_cast<void**>(&d_scores), scores_bytes)
        != cudaSuccess) {
        return;
    }

    // 阶段一：计算分数矩阵，每个线程负责一个 scores[i, j]。
    dim3 block_qk(16, 16);
    dim3 grid_qk(
        (seq_len + block_qk.x - 1) / block_qk.x,
        (seq_len + block_qk.y - 1) / block_qk.y
    );
    qk_scores<<<grid_qk, block_qk>>>(
        d_q, d_k, d_scores, seq_len, head_dim
    );

    // 阶段二：每个 block 处理一行，原地计算 row-wise softmax。
    // 这里按题目给定的配置启动。
    softmax_rows<<<seq_len, 128, 128 * sizeof(float)>>>(
        d_scores, seq_len, seq_len
    );

    // 阶段三：计算 out = P * V，每个线程负责一个 out[i, d]。
    dim3 block_pv(16, 16);
    dim3 grid_pv(
        (head_dim + block_pv.x - 1) / block_pv.x,
        (seq_len + block_pv.y - 1) / block_pv.y
    );
    pv_matmul<<<grid_pv, block_pv>>>(
        d_scores, d_v, d_out, seq_len, head_dim
    );

    // 同一默认 stream 中的 kernel 按顺序执行；释放临时缓冲区。
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics
    // 计算 exp(old_max - new_max)，将旧最大值基准下的累积量
    // 重缩放到 new_max 基准下。
    // 使用 expf 进行单精度浮点指数计算。
    return expf(old_max - new_max);
}

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

