# OpenAI Triton 简介（一）

<div style="text-align: center;">
    <img src="https://github.com/InfiniTensor/resources/blob/master/triton/images/triton-logo.png?raw=true" style="max-width: 25%; height: auto;">
</div>

感谢大家关注我们的公众号，如果您已经浏览过前面几篇文章，应该已经发现我们主要介绍一些 AI 领域底层的内容，比如如何使用 CUDA 编写各种算子，如何在寒武纪芯片上编程等等。今天的主角也与此相关，它是一门专注于深度学习领域的编程语言，对神经网络核函数的编写进行了抽象，使得高性能核函数的开发变得更加容易。与 C++、Rust、Python 等跨领域通用计算机语言不同，它是一门领域特定语言（以下简称 DSL），只专注于核函数的开发这一件事，但是也得益于此，它在这一方面非常在行。这位主角的名字就是：Triton。注意不要跟英伟达的推理服务器 Triton 搞混，这篇文章将要介绍的 Triton 是指目前 OpenAI 的 Triton，它们俩是重名。

掌握 CUDA 的小伙伴可能了解，想要极大限度地提高一个核函数的性能，我们要考虑许多因素，其中就包括以下四点：内存合并（memory coalescing）、共享内存管理（shared semory management）、流式多处理器内部的调度（scheduling within SMs）、流式多处理器（以下简称 SM）之间的调度（scheduling across SMs）。所以想要手工对核函数进行优化，通常就要求开发者对硬件架构有极其丰富的了解，对相应框架也要非常熟悉。即便如此，想要完成这样的优化，仍然需要花费许多时间精力，而这还只是针对英伟达的 GPU 来说。倘若我们需要支持多种硬件架构，则需要付出更多的努力才行。于是 Triton 应运而生。简单来讲，Triton 的编译器会帮助我们自动处理以上四点中的前三点，所以我们只需要关注 SM 之间的调度即可。下面我们通过一个例子（代码来自[官方教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)），来体会一下 Triton。

## 向量加法

先让我们来看一下向量加法，这几乎是最简单的核函数了，即便是使用 CUDA，相信也不会需要很多行，虽然这样显不出 Triton 的优越性，但是可以先体会一下 Triton 的语法。

### 开发核函数

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

浏览了这部分代码的小伙伴，可能会好奇：“这不是 Python 嘛？”没错，Triton 正是嵌在 Python 当中的一门 DSL，我们可以使用装饰器的形式来调用 Triton 的编译器，就像上面的 `@triton.jit`。简单来讲，Triton 的编译器可以通过这个装饰器拿到核函数的抽象语法树（AST），然后进行后续的操作，具体经过了怎样的过程，我们之后再具体讲。我们可以看到，代码中定义了一个 `add_kernel`，其中的参数 `x_ptr`、`y_ptr`、`output_ptr` 分别对应了我们想要相加的两个向量的指针，和一个输出向量的指针。没错，Triton 当中也有指针的概念，而且我们需要通过指针运算来确定想要访问的元素的位置。`n_elements` 是这三个向量的大小，而 `BLOCK_SIZE` 则表示我们想让每一个程序（program）处理多少元素，也就是一个分块（block）的大小。我们在这里缓一下思路，因为分块（block）和程序（program）是 Triton 当中最重要的概念，我们需要先进行理解。

简单来说，我们可以想象一个向量是一长根面条，分块就是把一根面条切成若干份小面条。我们同样可以把维度上升，比如矩阵就是张大饼，分块就是切出来的小饼。当然了，实际情况肯定比切面条和切饼复杂，比如怎么切，切多大，切出来的部分有什么顺序关系，但大家还是可以通过这样想象来进行理解。那么问题来了，我们为什么要分块呢？大饼好端端的，我切成小饼干啥？这就涉及到硬件的架构了，我们可以把 GPU 的运行理解成，一开始 CPU 会送来一张大饼，放在一张大桌子上，我们有一堆厨师，每一位厨师都要处理大饼的一部分，但他们得在自己的房间才有厨具处理，所以想要处理，就得有人从大饼上切下来这部分，送到厨师的房间，厨师放在自己的小桌子上加工，加工完再叫人送回去，但厨师不知道他们要做什么菜，只能听从厨师长的吩咐，所以每进行一步工序都得如此。如果只有一步工序，反正也只来回送两次，倒是没什么，但如果工序多了起来，那么反复送饼就会浪费很多时间，倒不如厨师长先分好饼，整明白啥工序可以合在一起给一位厨师完成，然后就能放在厨师的小桌子上，让尽量多的工序一次性完成，减少往返的次数，这样就提高了效率。而那张大桌子就是 GPU 的全局内存（global memory），小桌子则是 GPU 的共享内存（shared memory），而厨师就是程序（program）。理解了这些，现在就让我们看看函数体里都有什么吧。

<div style="text-align: center;">
    <img src="https://github.com/InfiniTensor/resources/blob/master/triton/images/gpu-kitchen.jpg?raw=true" style="max-width: 50%; height: auto;">
</div>

首先第一行是 `pid = tl.program_id(axis=0)`，这一行很简单，就是搞明白我们是哪一位厨师，毕竟这关系到了我要处理哪部分饼。接下来的两行就是在计算 `offsets`，以便于后面定义 `mask` 和定位指针。大家注意 `offsets` 的定义中有一项 `tl.arange(0, BLOCK_SIZE)`，这是因为一个分块有这么多个元素，我们都需要进行计算，第一个元素是 `block_start + 0`，而最后一个元素是 `block_start + BLOCK_SIZE - 1`， 所以 `offsets` 才是复数形式，它是整个分块每个元素的 `offsets`。但要注意的是，为了使这个核函数能够给任意大小的向量使用，我们必须再提供一个 `mask`，因为有些形状的向量，分块之后，最后的一个分块可能不足 `BLOCK_SIZE` 个，这时候我们就得忽略掉缺少的部分才行，所以我们只计算 `offsets < n_elements` 的部分，这就是 `mask`。好了，有了这些准备工作，剩下的内容就没什么了，我们先是从指针指向的位置把 `x` 和 `y` 当前分块的元素都 `load` 进来，加在一起，之后再 `store` 进 `output`，就完成了对这个分块的操作。由于参数张量被分成的每一块都被执行了这样的操作，因此即便对于整体而言，加法也被完成了。

### 调用核函数

好，现在我们写好了核函数，我们该如何调用呢？这个简单，由于 Triton 支持 PyTorch，所以我们可以直接把 `torch.Tensor` 传给核函数，就像下面这样。

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

在加入一些 `assert` 的限制条件，和 `output` 这种返回值分配后，我们就开始实际调用我们写好的 `add_kernel`。其中的 `grid` 就跟 CUDA 函数当中 `<<<...>>>` 形式的执行配置一样，用来定义如何进行并行，而这个配置在 Triton 当中就很好理解，就是单纯确定我们需要多少个程序，对于向量加法就更简单了，既然一个程序处理 `BLOCK_SIZE` 这么多元素，总共有 `n_elements` 这么多的元素，那我除一下不就知道需要多少程序了，唯一需要注意的就是，`n_elemtns` 可能不是 `BLOCK_SIZE` 的整数倍，所以我们得向上取整，才能保证没有遗漏，这也是为什么我们使用了 `triton.cdiv`，它是向上取整的除法（ceiling division）的意思。最后就是 `BLOCK_SIZE`，我们手动传入了 `1024`，意思就是我们想让一个程序处理 `1024` 个元素。这样，一个可以相加两个一维 `torch.Tensor` 的 `add` 函数就完成了。值得一提的是，这个 `BLOCK_SIZE` 不是必须要人为提供，只是作为第一个例子，这种形式更方便大家理解，在本系列后面的文章中，我们会讲如何让 Triton 进行 autotune，这样我们就不需要人为确定一个具体的值了。

如果想要进行实际运行，可以执行以下代码：

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

如果代码没有问题，应该会有如下输出，这就代表我们的结果跟 PyTorch 内置的加法所输出的结果一致：

```
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
The maximum difference between torch and triton is 0.0
```

好了，本文到这里就结束了，希望大家能通过这个简单的例子对 Triton 有一个初步的认知。本系列之后的文章还会介绍如何使用 Triton 实现更复杂的核函数，届时可能大家对 Triton 的易用性就会有更多的了解。再次感谢大家的关注！

## 引用

- [Introducing Triton: Open-source GPU programming for neural networks](https://openai.com/index/triton/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
