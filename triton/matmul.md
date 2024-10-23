# OpenAI Triton 简介（二）

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/wechat/refs/heads/master/triton/images/triton-logo.png" style="max-width: 25%; height: auto;">
</div>

感谢大家关注我们的公众号，这是我们《OpenAI Triton 简介》系列的第二篇文章。在上一篇文章里，我们已经简单地介绍了 Triton，并且讲解了一个由 Triton 写成的向量加法计算内核。当然，由于向量加法在任何编程模型里都很简单，Triton 的优越性可能并没有得到良好的展现。接下来，让我们一起来看看如何使用 Triton 实现高性能的矩阵乘法。注：本文代码主要来自 Triton 的[官方教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。

## 矩阵乘法

相信大家对矩阵乘法都很熟悉，但是这里还是给出它的公式和图示，方便下面的讲解：


$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

<div style="text-align: center;">
    <a title="File:Matrix multiplication diagram.svg:User:Bilou See below., CC BY-SA 3.0 &lt;http://creativecommons.org/licenses/by-sa/3.0/&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Matrix_multiplication_diagram_2.svg">
        <figure>
            <img width="256" alt="Multiplication of 2*4 and 3*2 matrices, giving intuition to the linear algebra." src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Matrix_multiplication_diagram_2.svg/256px-Matrix_multiplication_diagram_2.svg.png?20131021231827">
            <figcaption>File:Matrix multiplication diagram.svg:User:BilouSee below., CC BY-SA 3.0 <http://creativecommons.org/licenses/by-sa/3.0/>, via Wikimedia Commons</figcaption>
        </figure>
    </a>
</div>

### 开发核函数

温馨提示：下面的代码很长，所以一时间摸不到头脑是非常正常的，请不要钻牛角尖，简单浏览一下后请直接跳转至代码末尾继续阅读。

```python
import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
```

好，到这里，大家应该简单地过了一遍以上代码。我们可以把上面的代码分成三部分：autotune config 定义、矩阵乘法计算内核定义、Leaky ReLU 计算内核定义。有了上一篇文章对向量加法计算内核编写的介绍，其实最后一部分就变得非常好理解，就是简单地实现了一个 Leaky ReLU 函数：当 $x >= 0$ 的时候，返回 $x$，要不然返回 $\text{negative\_slope} \times x$（这里的 `negative_slop` 是 $0.01$）。

$$
  \text{LeakyReLU}(x) = 
  \begin{cases} 
  x, & \text{if } x \geq 0 \\
  \text{negative\_slope} \times x, & \text{otherwise} 
  \end{cases}
$$

接下来主要让我们一起看一下第二部分：矩阵乘法计算内核的定义。在正式看 Triton 实现之前，我们先来看一下所要实现的算法的伪代码：

```
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

这套算法的原理就是把 $A$、$B$、$C$ 三个矩阵（其中 $A$ 的大小是 $(M, K)$，$B$ 的大小是 $(K, N)$），按照 `BLOCK_SIZE_M`、`BLOCK_SIZE_N` 、`BLOCK_SIZE_K`三个大小，切成若干块，再进行计算。这样做的好处就是，我们可以将 `m` 和 `n` 两个 `for` 循环在 GPU 上进行并行，从而充分利用 GPU 善于并行的特质。大家可以结合以下这张示意图进行理解。

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/wechat/refs/heads/master/triton/images/tiled-matmul.png">
</div>

如图所示，每个 $C$ 的分块，都对应一行 $A$ 的分块和一列 $B$ 的分块。我们从左到右遍历 $A$ 中的那行分块，同时从上到下遍历 $B$ 中的那列分块，使得 $A$ 和 $B$ 的分块一一对应，遍历的过程中把分块之间矩阵乘法的结果累积起来，最终的结果存入对应的 $C$ 的分块中。

理解了这套算法后，实现起来就很直接了。唯一的难点，就是使用指针运算，定位所要操作的分块的位置。对于一个矩阵 `x`，`x[i][j]` 在内存中的位置可以通过 `x_ptr + i * x_stride_i + j * x_stride_j` 来得出。那么一个分块 `x[m : m + BLOCK_SIZE_M, n : n + BLOCK_SIZE_N]` 所对应的伪代码就是：

```
x_ptr + (m : m + BLOCK_SIZE_M)[:, None] * x.stride(0) + (n: n + BLOCK_SIZE_N)[None, :] * x.stride(1)
```

这样的话，我们就可以把指向 $A$ 那一行分块和 $B$ 那一列分块的指针，表示为如下形式：

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
```

前两行最后的 `%` 是为了确保在 `M` 不是 `BLOCK_SIZE_M` 的整数倍的时候，还依然可以得到正确的结果。

还记得算法中遍历 `k` 的 `for` 循环吧，在每轮循环最后，我们可以像下面这样更新这些指针：


```python
a_ptrs += BLOCK_SIZE_K * stride_ak
b_ptrs += BLOCK_SIZE_K * stride_bk
```

好，那么问题来了，我们进行如上初始化的前提是有 `pid_m` 和 `pid_n`。那我们该如何使用 `tl.program_id` 来拿到它们呢？有一种做法是：


```python
pid = tl.program_id(axis=0)
grid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

这样的话，矩阵乘法的并行计算就会如下图所示：

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/wechat/refs/heads/master/triton/images/row-major-ordering.png">
</div>

假如我们所计算的 $A$、$B$、$C$ 都被分成了 $9$ x $9$ 这么多的块，那么每计算出一行 $C$ 的分块，也就是每写入 $9$ 个 $C$ 的分块，我们都需要读取 $9$ 个 $A$ 的分块和 $81$ 个 $B$ 的分块，也就是总共读取 $90$ 个分块。

我们还可以尝试按照如下做法来初始化 `pid_m` 和 `pid_n`：

```python
# Program ID
pid = tl.program_id(axis=0)
# Number of program ids along the M axis
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# Number of programs ids along the N axis
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# Number of programs in group
num_pid_in_group = GROUP_SIZE_M * num_pid_n
# Id of the group this program is in
group_id = pid // num_pid_in_group
# Row-id of the first program in the group
first_pid_m = group_id * GROUP_SIZE_M
# If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# *Within groups*, programs are ordered in a column-major order
# Row-id of the program in the *launch grid*
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# Col-id of the program in the *launch grid*
pid_n = (pid % num_pid_in_group) // group_size_m
```

这样的话，矩阵乘法的并行计算就会如下图所示：

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/wechat/refs/heads/master/triton/images/grouped-ordering.png">
</div>

可以看出，现在每写入 $9$ 个 $C$ 的分块，我们就只需要读取 $27$ 个 $A$ 的分块和 $27$ 个 $B$ 的分块，也就是总共读取 $54$ 个分块，减少了总共的内存访问，从而可以提升计算内核的性能。而这也正是 `matmul_kernel` 最开始那部分代码的来历。

理解了这部分代码之后，接下来的内容就不难了：


```python
# -----------------------------------------------------------
# Iterate to compute a block of the C matrix.
# We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
# of fp32 values for higher accuracy.
# `accumulator` will be converted back to fp16 after the loop.
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Load the next block of A and B, generate a mask by checking the K dimension.
    # If it is out of bounds, set it to 0.
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    # We accumulate along the K dimension.
    accumulator = tl.dot(a, b, accumulator)
    # Advance the ptrs to the next K block.
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

我们先是定义了一个大小与 $C$ 的分块相同的累加器 `accumulator`，用来存储和累加中间结果，之后遍历 $A$ 的行分块和 $B$ 的列分块，依次进行矩阵乘法，并把结果累加在 `accumulator`。唯一需要注意的就是 `tl.load` 中的 `mask` 和 `other`，它们存在的目的就是确保我们只读取 $A$ 和 $B$ 里面的东西，如果超出了边界，就把那部分设为 $0$。

很多时候，矩阵乘法后面会紧跟一些其它的操作，比如上面提到的 $\text{LeakyReLU}$，如果能够把这样的操作融合进同一个计算内核，就能够进一步提高性能，而这样的算子融合，正是 Triton 所擅长的。

```python
# You can fuse arbitrary activation functions here
# while the accumulator is still in FP32!
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)
```

我们只需要 `accumulator = leaky_relu(accumulator)` 就可以完成融合，可谓是十分简便。

代码中最后的部分也很简单：

```python
# -----------------------------------------------------------
# Write back the block of the output matrix C with masks.
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

这部分在做的事情，就是通过指针运算，锁定 $C$ 分块的位置，之后再把结果写入其中。至此，`matmul_kernel` 函数体内的部分就都讲解完成了。

最后让我们来讲解一下 autotune config 定义，这部分内容，在上一篇文章讲解向量加法的时候是没有的，那为什么在这里就有了呢？如果大家回忆一下，就会发现，在调用 `add_kernel` 时，我们传入了一个 `BLOCK_SIZE=1024`，而这里的 `1024`，是我们人工选择的。而正因为我们人工选择了 `BLOCK_SIZE`，所以上次的时候，就没有引入 autotune config。但是人工选择，并不是我们想要的，因为对于不同架构不同平台，这样的元参数的选择，可能往往是繁杂的，我们几乎不可能每种架构每个平台都提前选择好，而这时，autotune 就显得很重要了。我们提前创建好不同平台架构下的配置，并且通过装饰器 `triton.autotune` 传入，这样，编译器就可以根据这些配置进行自动调优。

可以看到，我们代码开始的地方，定义了两组 autotune config，一组是 CUDA 的，另一组是 HIP 的，我们通过 `is_cuda` 来判断我们目前是不是运行在 CUDA 上，并且通过 `get_autotune_config` 来抓取相应的配置列表。每一组配置列表都由若干 `triton.Config` 组成，自动调优，本质上就是在选择使用哪一组 `triton.Config`。

```python
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
```

在 `triton.autotune` 中，我们先是传入了配置，之后又传入了 `key=['M', 'N', 'K']`，那它们又是什么意思呢？我们要清楚，Triton 目前的运行模式是 JIT 编译，也就是即时编译，而自动调优也是即时发生的。所以为了保证自动调优的效果，我们需要告诉 Triton 的编译器，当哪些传入计算内核的参数变动时，我们应当重新进行自动调优。这里，我们的意思就是，如果传入的参数中的 `M`、`N`、`K` 改变了，就应当重新进行自动调优。

到这里，矩阵乘法计算内核的主要内容就都讲解完毕了。

### 调用核函数

接下来，就让我们来看一下，这个 `matmul_kernel` 该怎样被调用。

```python
def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c
```

这部分跟向量加法的调用相差不大，首先我们使用 `assert` 定义了一些限制条件，再创建了 `c` 这样一个张量来存储结果。之后，我们定义了 `grid`，也就是 Triton 的执行配置。这里，我们 `grid` 的大小是：

$$
\lceil M \div BLOCK\_SIZE\_M \rceil \times \lceil N \div BLOCK\_SIZE\_N \rceil
$$

也就是说，我们将会发射这么多的 programs，每一个 program 都会并行执行我们所定义的 `matmul_kernel`。

如果想要进行实际运行，可以执行以下代码：

```python
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
```

如果代码没有问题，应该会有如下输出，这就代表我们的结果跟 PyTorch 内置的加法所输出的结果一致：

```
triton_output_with_fp16_inputs=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp16_inputs=tensor([[-10.9531,  -4.7109,  15.6953,  ..., -28.4062,   4.3320, -26.4219],
        [ 26.8438,  10.0469,  -5.4297,  ..., -11.2969,  -8.5312,  30.7500],
        [-13.2578,  15.8516,  18.0781,  ..., -21.7656,  -8.6406,  10.2031],
        ...,
        [ 40.2812,  18.6094, -25.6094,  ...,  -2.7598,  -3.2441,  41.0000],
        [ -6.1211, -16.8281,   4.4844,  ..., -21.0312,  24.7031,  15.0234],
        [-17.0938, -19.0000,  -0.3831,  ...,  21.5469, -30.2344, -13.2188]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
triton_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  28.7031,   8.6719, -40.7500],
        [ 10.0000,  37.0000,  -5.5664,  ...,  20.9844,  46.8125,  30.8281],
        [ 19.5625,  -3.0078, -20.0469,  ...,  -2.1309,  -8.0625,  12.5625],
        ...,
        [-18.1562, -34.1562, -27.4219,  ..., -27.3906, -24.0938, -12.3516],
        [ -3.3945,  -8.6250, -23.6562,  ...,  -4.1094,  -3.5332, -16.0781],
        [-23.9688,  -3.2637, -33.6875,  ...,  17.3125, -36.6250,  25.8594]],
       device='cuda:0', dtype=torch.float16)
torch_output_with_fp8_inputs=tensor([[-21.4375,  13.1719,   6.0352,  ...,  28.7031,   8.6719, -40.7500],
        [ 10.0000,  37.0000,  -5.5664,  ...,  20.9844,  46.8125,  30.8281],
        [ 19.5625,  -3.0078, -20.0469,  ...,  -2.1309,  -8.0625,  12.5625],
        ...,
        [-18.1562, -34.1562, -27.4219,  ..., -27.3906, -24.0938, -12.3516],
        [ -3.3945,  -8.6250, -23.6562,  ...,  -4.1094,  -3.5332, -16.0781],
        [-23.9688,  -3.2637, -33.6875,  ...,  17.3125, -36.6250,  25.8594]],
       device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
```

好了，对 Triton 实现矩阵乘法的讲解到这里就结束了。通过以上内容，大家可以发现，Triton 实现某一算法的重点，就在于分块的确定，即通过 `BLOCK_SIZE_*` 等元参数，结合 `program_id` 进行指针运算，从而定位到相应的分块，但是这部分操作，却几乎并不涉及更底层的硬件细节。希望大家能通过这个例子体会到 Triton 的优势。再次感谢大家的关注！

## 引用

- [Introducing Triton: Open-source GPU programming for neural networks](https://openai.com/index/triton/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
