# 九齿与面向张量的元编程

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/ninetoothed-logo.png" style="max-width: 50%; height: auto;">
</div>

感谢大家对我们的关注。今天我想跟大家聊一聊我们的一项工作——九齿，以及它背后的逻辑——面向张量的元编程。

## 张量

如果大家对深度学习领域有一定了解的话，对张量这一概念一定也不陌生。一般来讲，在深度学习领域里，张量是指用于表示数据的多维数组，是模型进行计算的基本数据结构。举例来说，我们所熟悉的标量、向量、矩阵，都是张量，分别是零维张量、一维张量、二维张量。常见的深度学习框架，如 [PyTorch](https://pytorch.org/)、[TensorFlow](https://www.tensorflow.org/) 等，都把张量作为主要的数据结构。

有数据，就可以操作数据，对于张量而言也是如此，我们可以做像张量创建、张量计算、张量变换等多种操作。以 PyTorch 为例，我们可以用 `torch.empty` 创建一个空的张量，可以用 `torch.mm` 做矩阵乘法，还可以用 `torch.transpose` 做张量转置。如果您对张量尚不熟悉，可以参考这篇 [PyTorch 的教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。

## 元编程

那么，什么叫元编程呢？这里给出一段来自维基百科的描述。

> 元编程，是指某类计算机程序的编写，这类计算机程序编写或者操纵其它程序（或者自身）作为它们的资料，或者在编译时完成部分本应在运行时完成的工作。多数情况下，与手工编写全部代码相比，程序员可以获得更高的工作效率，或者给与程序更大的灵活度去处理新的情形而无需重新编译。

像是 C++ 当中，我们就可以利用模板进行元编程，也就是大家熟悉的模板元编程。

现在大家分别了解了张量和元编程，肯定就会好奇：这两者会有什么联系？什么叫面向张量的元编程？为了回答这些问题，接下来就让我们来正式介绍一下我们的项目——九齿。

## 九齿

目前，[九齿](https://github.com/InfiniTensor/ninetoothed)是一门基于 [Triton](https://triton-lang.org/) 的领域特定语言（DSL），旨在进一步简化高性能计算内核的开发。它通过引入面向张量的元编程，抽象掉了指针算术运算和内存访问等底层细节，能够降低并行编程的门槛。九齿能够让开发者使用少量简洁的代码实现较高性能的计算内核，并且可以提高代码的可读性和可维护性。

用更简洁一点的话来说，九齿跟 Triton 都是用来干一种活的，就是写深度学习领域的计算内核，只是九齿在 Triton 的基础上做了进一步的抽象，所以写起来更简单，写出来的代码更易读了。那么啥是计算内核呢？简单来讲，就是那些在 GPU 之类的加速卡上运行的，做计算的程序，像上文中提到的 `torch.mm`，底层调用的就是某个计算内核。如果大家希望对 Triton 有一些更多的了解，也欢迎大家来阅读我们[《OpenAI Triton 简介》](https://github.com/InfiniTensor/resources/tree/master/triton)系列的文章。

但是光说没用，所以这里先给大家放一段使用九齿实现矩阵乘法的代码，先带大家感受一下，之后我们会慢慢展开讲解。

```python
BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


def arrangement(input, other, output):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator


kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))
```

可以看出，整个实现当中，并没有出现任何显式的指针运算或是内存访问，而是基本以描述式的函数调用为主。当然了，我们肯定还是得先了解一些九齿的核心概念，才能完全明白这段代码干了些什么。

### 核心概念

那么接下来，就让我们一起来看一下这些核心概念。

#### 符号张量

与很多深度学习框架类似，张量在九齿当中是最为核心的概念，但是九齿当中的张量，与其他框架中的有些许不同，它们并不存储实际数据，仅在 `shape`、`strides` 等成员变量中存储符号表达式，所以我们可以叫它们符号张量。在九齿中，我们可以使用 `Tensor` 来创建一个张量。如下方代码所示，`Tensor(2)` 表示构造一个二维张量，也就是一个矩阵。我们可以看到，这个张量的 `shape` 成员里所存储的，都是符号，而非具体的数值：

```
>>> from ninetoothed import Tensor
>>> x = Tensor(2)
>>> x.shape
(ninetoothed_tensor_0_size_0, ninetoothed_tensor_0_size_1)
```

#### 符号

符号这一概念，与[这篇 SymPy 教程](https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html)当中描写的类似。符号并不存储实际的数值，只存储符号或是符号表达式，所以允许进行一些符号化的数学运算。在九齿中，我们可以使用 `Symbol` 来创建一个符号。例如，在下面的代码里，我们先是创建了名为 `BLOCK_SIZE_M` 和 `BLOCK_SIZE_N` 的两个符号，之后对它们进行了乘法操作：

```
>>> from ninetoothed import Symbol
>>> BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M")
>>> BLOCK_SIZE_M
BLOCK_SIZE_M
>>> BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N")
>>> BLOCK_SIZE_N
BLOCK_SIZE_N
>>> BLOCK_SIZE_M * BLOCK_SIZE_N
BLOCK_SIZE_M * BLOCK_SIZE_N
```

#### 面向张量的元编程

得益于符号张量，我们可以对九齿中的张量进行一些编译期操作，这样的操作被称为元操作，如 `tile`、`expand`、`squeeze`、`permute` 等。例如，在这一段代码中，我们对 `x` 进行了 `tile` 操作，即把 `x` 分为形状为 `(BLOCK_SIZE_M, BLOCK_SIZE_N)` 的块：

```
>>> x_tiled = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
>>> x_tiled.shape
((ninetoothed_tensor_0_size_0 - (BLOCK_SIZE_M - 1) - 1 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + 1, (ninetoothed_tensor_0_size_1 - (BLOCK_SIZE_N - 1) - 1 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N + 1)
>>> x_tiled.dtype.shape
(BLOCK_SIZE_M, BLOCK_SIZE_N)
```

我们注意到，`x_tiled` 的 `dtype` 也有 `shape` 这一成员变量。这是由于，九齿当中的张量是可以嵌套的，即一个张量的元素也可以是一个张量。也就是说，在 `tile` 的过程中，我们创建了一个双层的张量，其中外层张量的每一个元素，都是一个内层张量。为了方便理解，我们可以使用如下的数值示例来进行说明：

```
>>> BLOCK_SIZE_M = 2
>>> BLOCK_SIZE_N = 2
>>> x = Tensor(shape=(4, 8))
>>> x_tiled = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
>>> x_tiled.shape
(2, 4)
>>> x_tiled.dtype.shape
(2, 2)
```

就像下图所示的那样，我们所做的，是把一个形状为 `(4, 8)` 的张量 `x` 分成了形状为 `(2, 2)` 的块（内层张量），总共分成了 `(2, 4)` 个这样的张量（外层张量）：

![x-tiled.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/x-tiled.png)

#### 排布与应用范式

介绍完了元操作，大家应该就可以理解，我们在编译期，对单个张量可以进行怎样的操作。我们把一系列这样的操作，称之为排布。但是这还不够，因为我们还需要建立多个参数张量之间的联系。

这样的联系，在九齿中，是由编译器进行的：九齿的编译器会根据各个参数张量排布后的最外层张量的形状启动程序，并把次外层张量映射到这些程序上。

我们可以通过一个简单的排布函数，来理解这句话：

```python
def arrangement(x, y, z, BLOCK_SIZE=ninetoothed.block_size()):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,))
```

在这个函数当中，我们分别对 `x`、`y`、`z` 三个向量进行了 `tile` 操作，想要把每个向量分成 `BLOCK_SIZE` 这么大的块。假如每个向量的长度为 `16`，`BLOCK_SIZE` 为 `2`，那么每个向量就可以被分成 `8` 块，每一块的长度就是 `2`，则排布后的 `x`、`y`、`z` 就可以分别如下图所示：

![x-arranged.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/x-arranged.png)

![y-arranged.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/y-arranged.png)

![z-arranged.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/z-arranged.png)

那么，根据这样的排布，九齿的编译器便会启动 `8` 个程序，并把排布后的 `x`、`y`、`z` 的最外层张量的每个元素，也就是次外层张量，与这 `8` 个程序一一对应。

很好，现在我们有了这些对应关系，也能够相应地启动程序。但是我们距离能够完整地实现一个算法，还仍然差一步，因为我们还没有定义每个程序要做的事情。换句话说，我们还需要定义如何应用排布后的张量。在九齿中，我们可以通过定义应用函数的方式，来做到这一点。

以向量加法为例，我们可以定义如下的应用函数：

```python
def application(x, y, z):
    z = x + y
```

代码逻辑很简单，就是把 `x` 和 `y` 相加，并把结果放入 `z` 中。但是需要注意的是：应用函数的参数，是参数张量排布后的最外层张量的元素，也就是次外层张量，而不是张量本身。也就是说，如果套用上面的假设，这里的 `x`、`y`、`z` 都是指长度为 `2` 的块，而不是长度为 `16` 的原本的张量。

到了这里，我们已经定义好了排布函数和应用函数，剩下的就是把它们二者整合成一个计算内核。在九齿中，我们可以使用 `ninetoothed.make` 来进行这个整合：

```python
kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))
```

这段代码的意思就是说，我想要按照 `arrangement` 函数对三个一维张量，也就是向量，进行排布，并按照 `application` 函数应用排布后的张量，最终形成一个计算内核 `kernel`。我们把这样构造计算内核的范式，称之为排布与应用范式。

我们可以如下所示对 `kernel` 进行调用：

```python
import torch

dtype = torch.float16
device = "cuda"

x = torch.tensor((1, 2, 3), dtype=dtype, device=device)
y = torch.tensor((4, 5, 6), dtype=dtype, device=device)

z = torch.empty_like(x)
kernel(x, y, z)

reference = torch.tensor((5, 7, 9), dtype=dtype, device=device)
assert torch.allclose(z, reference)
```

可以看到，我们在调用 `kernel` 时，并没有提供 `BLOCK_SIZE` 的实际取值。这是因为，在构造 `BLOCK_SIZE` 时，我们使用了 `ninetoothed.block_size`，这代表我们希望使用九齿编译器生成的配置来进行自动调优。如果我们希望人为提供取值（比如我们在进行调试时），我们可以直接使用具体数值来进行赋值：

```python
def arrangement(x, y, z, BLOCK_SIZE=1024):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,))
```

#### 索引和迭代

通过向量加法，我们简单地了解了一下使用九齿开发计算内核的流程。在向量加法中，参数张量经过排布变成了双层的张量，但是九齿当中的张量并不局限于双层，也可以是三层甚至更多层，不过只有排布后的最外层会被用于启动程序。换句话说，三及以上层的张量，在应用函数里，也是层级张量，是可以被索引和迭代的。

现在，是时候让我们回到上面提到的矩阵乘法，通过实现一个矩阵乘法计算内核来理解一下九齿中的的索引和迭代，并进一步体会排布与应用范式。

在正式开始之前，我们得先知道，我们想要实现的是一个怎样的算法。这里是一个该算法的图示：

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/triton/images/tiled-matmul.png">
</div>

简单来讲，我们要对三个矩阵进行分块，之后对于 $C$ 中的每个块，我们都需要迭代 $A$ 中对应的那行块以及 $B$ 中对应的那列块，并把每次迭代中的 $A$ 的块与 $B$ 的块进行一个小规模的矩阵乘法，这样应当写入这个 $C$ 的块的值，就可以通过累加这些小规模矩阵乘法的结果来得到。

好，现在我们有了算法的描述，就可以正式开始实现了，还是让我们从排布开始：

```python
BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


def arrangement(input, other, output):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged
```

在代码中，我们首先定义了 `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K` 三个符号，用于表示分块的形状。具体来讲，我们先把 `output` 矩阵 `tile` 成形状为 `(BLOCK_SIZE_M, BLOCK_SIZE_N)` 的块，把 `input` 矩阵 `tile` 成形状为 `(BLOCK_SIZE_M, BLOCK_SIZE_K)` 的块，再把 `other` 矩阵 `tile` 成形状为 `(BLOCK_SIZE_K, BLOCK_SIZE_N)` 的块：

![input-arranged-0.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/input-arranged-0.png)

![other-arranged-0.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/other-arranged-0.png)

![output-arranged-0.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/output-arranged-0.png)

我们注意到，只进行分块对于矩阵乘法是不足的。按照上面的算法图示，`output` 当中的每个块，对应的是 `input` 的一行块，和 `other` 的一列块，所以我们还需要对 `input` 和 `other` 进行进一步的 `tile`，也就是把 `input` 的每一行 `tile` 在一起，以及把 `other` 的每一列 `tile` 在一起：

![input-arranged-1.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/input-arranged-1.png)

![other-arranged-1.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/other-arranged-1.png)

但是到这里也还没有结束。还记得九齿的编译器是如何建立多个参数张量之间的联系的嘛？我们在这里再来回顾一下：

> 九齿的编译器会根据各个参数张量排布后的最外层张量的形状启动程序，并把次外层张量映射到这些程序上。

这句话为什么重要呢？因为我们可以由此引申出一条重要的推论：各个参数张量排布后的最外层张量应当具有相同的形状。

很明显，目前我们的三个参数张量排布后的最外层张量，形状并不相同，分别为 `(4, 1)`、`(1, 4)`、`(4, 4)`。这往往说明我们的排布并不正确，或者尚未完成。通过图示我们可以知道，我们需要把 `input` 的每一行块，与 `other` 的每一列块对齐，这一点我们可以通过 `expand` 来做到，也就是把 `input` 沿着横向 `expand`，把 `other` 沿着竖向 `expand`，均 `expand` 至与 `output` 有相同的形状：

![input-arranged-2.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/input-arranged-2.png)

![other-arranged-2.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/other-arranged-2.png)

至此，三个参数张量排布后的最外层张量，便具有了相同的形状。其实，排布阶段可以到此为止，因为我们已经可以据此写出应用函数，但是我们发现，刚才所分成的 `input` 的行块和 `other` 的列块是二维的，并且具有 `(1, ...)` 和 `(..., 1)` 这样形式的形状。也就是说，如果不进行其他操作，那么我们索引行块和列块的方式就得是 `input[0, k]` 和 `other[k, 0]`；如果我们想要基于 `input` 找到 `k` 的范围，那就需要使用 `input.shape[1]`。但是我们知道，大小为 `1` 的维度，在这种情况下完全可以被去掉，这就是为什么我们在最后加入了 `squeeze` 操作：

![input-arranged-3.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/input-arranged-3.png)

![other-arranged-3.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/other-arranged-3.png)

这样，我们在索引行块和列块时就可以使用 `input[k]` 和 `other[k]` 了；寻找 `k` 的范围时也可以使用 `input.shape[0]` 了。

到这里，整个排布阶段就告一段落了。排布的最终结果如下所示：

![input-arranged-3.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/input-arranged-3.png)

![other-arranged-3.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/other-arranged-3.png)

![output-arranged-0.png](https://raw.githubusercontent.com/InfiniTensor/resources/refs/heads/master/ninetoothed/images/output-arranged-0.png)

现在让我们来看应用函数：

```python
def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator
```

在函数体中，我们先定义了一个 `accumulator`，用于累加中间结果，之后就迭代了对应好的 `input` 的行块和 `other` 的列块，并把小规模矩阵乘法的结果累加到了 `accumulator` 当中，最后再把 `accumulator` 写入了对应的 `output` 的块当中。由于参数张量被分成的每一块都被执行了这样的操作，因此对于整体而言，矩阵乘法就完成了。

与向量加法相同，在定义好 `arrangement` 和 `application` 后，我们可以使用 `ninetoothed.make` 对它们进行整合，从而构造一个可以运行的 `kernel`：

```python
kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))
```

我们可以如下所示对 `kernel` 进行调用：

```python
import torch

dtype = torch.float16
device = "cuda"

lhs = torch.tensor(((1, 2), (3, 4)), dtype=dtype, device=device)
rhs = torch.tensor(((5, 6), (7, 8)), dtype=dtype, device=device)

output = torch.empty((lhs.shape[0], rhs.shape[1]), dtype=dtype, device=device)
kernel(lhs, rhs, output)

reference = torch.tensor(((19, 22), (43, 50)), dtype=dtype, device=device)
assert torch.allclose(output, reference)
```

这些就是九齿当中最核心的几个概念。

好了，本文到这里就结束了，希望大家能通过这篇文章对九齿有一个初步的认识。如果大家对九齿感兴趣，这里也有几个链接，供大家进一步的了解：

* GitHub 仓库：https://github.com/InfiniTensor/ninetoothed
* 九齿文档网站（有待完善）：https://ninetoothed.org/
* 一些九齿相关的例子：https://github.com/InfiniTensor/ninetoothed-examples
* 一个刚开始建设的九齿算子库：https://github.com/InfiniTensor/ntops

再次感谢大家的关注！

## 引用

本文使用了维基百科文章[元编程](https://zh.wikipedia.org/wiki/%E5%85%83%E7%BC%96%E7%A8%8B)中的素材, 该文章根据 [Creative Commons Attribution-Share-Alike License 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 创作。
