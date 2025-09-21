# Calculus basis for Machine Learning
[TOC]
## 常用导数公式与规则
# Commonly Used Derivatives

### Basic Rules
1. **Constant Rule**:
   $$\frac{d}{dx}[c] = 0$$
   (The derivative of a constant is always 0.)

2. **Power Rule**:  
$$\frac{d}{dx}[x^n] = n \cdot x^{n-1}$$

3. **Sum/Difference Rule**:  
$$\frac{d}{dx}[f(x) \pm g(x)] = \frac{d}{dx}[f(x)] \pm \frac{d}{dx}[g(x)]$$

4. **Constant Multiple Rule**:  
$$\frac{d}{dx}[c \cdot f(x)] = c \cdot \frac{d}{dx}[f(x)]$$


---

### Product and Quotient Rules  
1. **Product Rule**: $$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$  
2. **Quotient Rule**: $$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{g(x)^2} \quad (g(x) \neq 0)$$  

---

### 指数与对数函数  
1. **自然指数函数**: $$\frac{d}{dx}[e^x] = e^x$$  
2. **一般指数函数**: $$\frac{d}{dx}[a^x] = a^x \ln(a) \quad (a > 0, a \neq 1)$$  
3. **自然对数函数**: $$\frac{d}{dx}[\ln(x)] = \frac{1}{x} \quad (x > 0)$$  
4. **一般对数函数**: $$\frac{d}{dx}[\log_a(x)] = \frac{1}{x \ln(a)} \quad (x > 0, a > 0, a \neq 1)$$  

---

### Trigonometric Functions  
1. $$\frac{d}{dx}[\sin(x)] = \cos(x)$$  
2. $$\frac{d}{dx}[\cos(x)] = -\sin(x)$$  
3. $$\frac{d}{dx}[\tan(x)] = \sec^2(x)$$  
4. $$\frac{d}{dx}[\cot(x)] = -\csc^2(x)$$  
5. $$\frac{d}{dx}[\sec(x)] = \sec(x) \tan(x)$$  
6. $$\frac{d}{dx}[\csc(x)] = -\csc(x) \cot(x)$$  

---

### Exponential and Logarithmic Functions  
1. $$\frac{d}{dx}[e^x] = e^x$$  
2. $$\frac{d}{dx}[a^x] = a^x \ln(a) \quad (a > 0, a \neq 1)$$  
3. $$\frac{d}{dx}[\ln(x)] = \frac{1}{x} \quad (x > 0)$$  
4. $$\frac{d}{dx}[\log_a(x)] = \frac{1}{x \ln(a)} \quad (x > 0, a > 0, a \neq 1)$$  

---

### Inverse Trigonometric Functions  
1. $$\frac{d}{dx}[\arcsin(x)] = \frac{1}{\sqrt{1 - x^2}} \quad (|x| < 1)$$  
2. $$\frac{d}{dx}[\arccos(x)] = -\frac{1}{\sqrt{1 - x^2}} \quad (|x| < 1)$$  
3. $$\frac{d}{dx}[\arctan(x)] = \frac{1}{1 + x^2}$$  

---

### Hyperbolic Functions  
1. $$\frac{d}{dx}[\sinh(x)] = \cosh(x)$$  
2. $$\frac{d}{dx}[\cosh(x)] = \sinh(x)$$  
3. $$\frac{d}{dx}[\tanh(x)] = \text{sech}^2(x)$$  
4. $$\frac{d}{dx}[\coth(x)] = -\text{csch}^2(x)$$  
5. $$\frac{d}{dx}[\text{sech}(x)] = -\text{sech}(x) \tanh(x)$$  
6. $$\frac{d}{dx}[\text{csch}(x)] = -\text{csch}(x) \coth(x)$$  

---

### Chain Rule  
1. **General Chain Rule**: $$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$  


##  1. Functions and Optimization
### 1.1 Review of concepts 
#### **Key Concepts:**

1. **Definition of a Function:**
   - A function is a relationship where an input \( x \) is mapped to an output \( y \), written as \( y = f(x) \).
   - The notation \( f: \mathbb{R}^D \to \mathbb{R} \) means:
     - \( f \) is a function.
     - It takes an input \( x \) from a domain (here, \( \mathbb{R}^D \), meaning a real vector space of \( D \)-dimensions).
     - It produces an output in \( \mathbb{R} \) (a real number).

2. **Key Properties of Functions:**
   - **Every input \( x \) corresponds to only one output \( y \).**
     - This is the defining property of a function: for each value of \( x \), there is only one value of \( y \).
   - **Independent and Dependent Variables:**
     - \( x \) is the **independent variable** because it is the input to the function.
     - \( y \) is the **dependent variable** because its value depends on \( x \).

3. **Example:**
   - The example provided is about distance traveled per hour (\( y \)) as a function of velocity (\( x \)).
     - For a given velocity \( x \), the distance traveled (\( y \)) is uniquely determined.

4. **Inputs and Outputs:**
   - The diagram at the bottom of the page illustrates the process:
     - **Input \( x \):** The value or object you provide to the function.
     - **Function \( f \):** The rule or process that transforms the input into an output.
     - **Output \( y \):** The result of applying the function to the input.

5. **Extended Concept:**
   - The thought bubble on the top right explains that the input \( x \) can be more than just a simple number:
     - \( x \) can be **vectors** (e.g., a list of numbers) or even **matrices** (e.g., a grid of numbers).
     - <font color='blue'> Similarly, the output \( f(x) \) can also be vectors or matrices, depending on the function. </font>

### 1.2 Optimization 
**Optimization of a Function**

The slide provides an overview of the concept of function optimization, explaining its mathematical basis and an example of its application in everyday life.


#### 1.2.1  **Key Points:**

1. **Definition of Function Optimization:**
   - A function \( f(\mathbf{x}) \) is defined, where the vector \( \mathbf{x} \) represents a set of variables \[ x_1, x_2, \dots, x_n \].
   - **Optimization** refers to finding the best \( \mathbf{x} \) that either:
     - **Maximizes** \( f(\mathbf{x}) \), or
     - **Minimizes** \( f(\mathbf{x}) \).
   - Mathematically, this is expressed as:
     \[
     f(\mathbf{x}) = f(x_1, x_2, \dots, x_n)
     \]

2. **Example of an Optimization Problem in Real Life:**
   - **Scenario:**
     - You have selected 3 classes for the semester.
     - Each class has a different effect on your GPA.
     - The goal is to **maximize your GPA** (the value of the function) by determining the optimal number of hours (the function variables) to spend on each class.
   - This involves using prior knowledge of how the number of hours impacts the GPA to find the best allocation of time.

#### 1.2.2 **Convex Function and Optimization**

This slide introduces the concept of convex functions and their significance in optimization problems. It explains the properties of convex functions and their role in simplifying optimization tasks.

##### **1. Convex Function**
- **Definition:**
  - A function \( f(\mathbf{x}) \) is **convex** if, for any two points \( \mathbf{x}_1 \) and \( \mathbf{x}_2 \), and for any \( \theta \) such that \( 0 \leq \theta \leq 1 \), the following inequality holds:
    \[
    f(\theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2) \leq \theta f(\mathbf{x}_1) + (1 - \theta) f(\mathbf{x}_2)
    \]
  - 函数图像在**任意两个点**之间的连线（权重平均值）始终在函数曲线的上方或与其重合。
  - This means that the value of the function at any point on the line segment between \( \mathbf{x}_1 \) and \( \mathbf{x}_2 \) is less than or equal to the weighted average of the function values at \( \mathbf{x}_1 \) and \( \mathbf{x}_2 \).

- **Key Property:**
  - A convex function has a "valley-like" shape, meaning it curves upwards and does not have multiple valleys or peaks.
<img src="p1.png" width="70%" height="70%" >




##### **2. Convex Optimization**
- **Definition:**
  - Convex optimization involves minimizing or maximizing a convex function \( f(\mathbf{x}) \).

- **Key Property:**
  - If \( f(\mathbf{x}) \) is convex, then:
    - **Every local minimum is also a global minimum.**
    - <font color='blue'>Convex functions have the property that any local minimum is a global minimum (last year mid term tested) </font>
  - This property makes optimization of convex functions significantly easier, as finding any local minimum guarantees the global minimum.
  - ###### **Local vs Global Optimization** 
    - Local optimization involves finding the best solution (local maximum or local minimum) within **<font color = 'blue'> a specific region of the function's domain.** </font>
    - Global optimization involves finding the absolute best solution (global maximum or global minimum) for a function over its **<font color = 'blue'>  entire domain.</font>**

- **Importance:**
  - Convex optimization is widely used in machine learning, economics, and engineering because it ensures efficient and reliable solutions.


#### 1.2.2 **Concave Function and Optimization**
##### Concave Function

A function \( f(x) \) is **concave** if, for any \( x_1, x_2 \in \text{domain of } f \), and \( \theta \in [0, 1] \), the following inequality holds:

\[
f(\theta x_1 + (1 - \theta)x_2) \geq \theta f(x_1) + (1 - \theta)f(x_2)
\]

This means that the function lies **above** or on the line segment connecting any two points on the graph of the function. Geometrically, the graph of a concave function appears "arched downward."



##### Concave Optimization

If \( f(x) \) is a **concave function**, then:

- **Every local maximum is also a global maximum.**

This is because a concave function has a single "peak" (or a flat region) and no other local maxima, making any local maximum also the global maximum.


---

##  2. Function Derivatives

### 2.1 Derivative of Univariate Functions 单变量

#### What are Derivatives?

The **derivative of \( f(x) \)** is the **slope of the tangent line** (instantaneous rate of change) at \( (x, f(x)) \).

\[
\frac{dy}{dx} = f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
\]

The process of calculating the derivatives of a function is called **differentiation**.


##### Example: \( y = x^2 \)

\[
\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{(x + \Delta x)^2 - x^2}{\Delta x}
\]




###### Differential Form:
The derivative can also be written as:
\[
dy = 2x \, dx
\]

- Here, \( dy \) is the **differential** (a small change in \( y \)) in terms of \( dx \) (a small change in \( x \)).


###### Applications:
- The differential form \( dy = 2x \, dx \) can be used to estimate the output difference \( \Delta y \) in terms of a small input difference \( \Delta x \).


### 2.2 Partial Derivative and Gradient

#### 2.2.1 梯度
##### **定义：**
梯度是一个向量，包含了函数对所有输入变量的偏导数。它将这些偏导数收集到一个行向量中。

\[
\nabla_{\mathbf{x}} f = \frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{\partial f(x)}{\partial x_1} & \cdots & \frac{\partial f(x)}{\partial x_n}
\end{bmatrix}
\]

- 梯度是一个 \( 1 \times n \) 的行向量，表示函数 \( f(x) \) 在每个输入方向上的变化率。

##### **关键点：**
- 梯度的作用是描述函数 \( f(x) \) 在输入空间（\(\mathbb{R}^n\)）中的变化方向和速率。
- 梯度向量的方向是函数增加最快的方向，向量的大小表示增加的速率。
#### 2.2.2 求偏导的计算规则


##### **1. 偏导数的规则适用性**
- PPT顶部的蓝色气泡中写道：
  > **“Many of the univariate function’s derivative rules hold for partial differentiation”**
  - 意思是：单变量函数的许多求导规则（如加法法则、乘法法则、商法则、链式法则和幂法则）同样适用于偏导数的计算。

- 右侧蓝框补充：
  > **“These include the sum rule, product rule, quotient rule, chain rule, and power rule.”**
  - 这些规则包括：加法法则、乘法法则、商法则、链式法则以及幂法则。


##### **2. 偏导数的规则**
###### **(1) 乘法法则（Product Rule）**
\[
\frac{\partial}{\partial x} \big(f(x) g(x)\big) = \frac{\partial f}{\partial x} g(x) + f(x) \frac{\partial g}{\partial x}
\]
- **解释：**
  - 对两个函数 \(f(x)\) 和 \(g(x)\) 的乘积求偏导时，结果是：
    1. \(f(x)\) 的偏导乘以 \(g(x)\) 本身；
    2. 加上 \(f(x)\) 本身乘以 \(g(x)\) 的偏导。


###### **(2) 加法法则（Sum Rule）**
\[
\frac{\partial}{\partial x} \big(f(x) + g(x)\big) = \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x}
\]
- **解释：**
  - 对两个函数之和求偏导时，结果是每个函数分别求偏导后相加。



###### **(3) 链式法则（Chain Rule）**
\[
\frac{\partial}{\partial x} g\big(f(x)\big) = \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}
\]
- **解释：**
  - 如果 \(g\) 是 \(f(x)\) 的函数（即复合函数），那么对 \(g(f(x))\) 求偏导时：
    1. 首先对 \(g\) 相对于 \(f\) 求偏导；
    2. 然后乘以 \(f\) 相对于 \(x\) 的偏导。

#### 2.2.3 线性代数和函数
##### 1. Jacobian Matrix
This slide provides a **review of the Jacobian matrix** for vector-valued functions. It explains the mathematical structure and meaning of the Jacobian matrix, particularly in the context of functions that map from \(\mathbb{R}^n\) to \(\mathbb{R}^m\). Here's a breakdown of its content:


 **Jacobian Matrix Definition**
- The Jacobian matrix, denoted as \(\mathbf{J}\) or \(\nabla_{\mathbf{x}} f\), represents the derivative of a **vector-valued function** \(f(\mathbf{x})\) with respect to the input vector \(\mathbf{x}\).
  
\[
\mathbf{J} = \nabla_{\mathbf{x}} f = \frac{\mathrm{d}f(\mathbf{x})}{\mathrm{d}\mathbf{x}}
\]

- The Jacobian is a matrix where:
  - **Rows** correspond to the partial derivatives of each output component (entries of \(\mathbf{f}(\mathbf{x})\)).
  - **Columns** correspond to the partial derivatives with respect to each input variable (entries of \(\mathbf{x}\)).

**Structure of the Jacobian Matrix**
The Jacobian matrix is written as:

\[
\mathbf{J} = 
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
\]

- **Rows:** Each row corresponds to one output function \(f_i(\mathbf{x})\), showing how it changes with respect to each input variable \(x_1, x_2, \dots, x_n\).
- **Columns:** Each column corresponds to one input variable \(x_j\), showing how all output components \(f_1, f_2, \dots, f_m\) change with respect to \(x_j\).


**Dimensions of the Jacobian**
- For a function \(f: \mathbb{R}^n \to \mathbb{R}^m\):
  - The **input** is an \(n\)-dimensional vector (\(\mathbf{x} \in \mathbb{R}^n\)).
  - The **output** is an \(m\)-dimensional vector (\(\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m\)).
  - Therefore, the Jacobian matrix \(\mathbf{J}\) has dimensions of \(m \times n\):
    - \(m\) rows (one for each output component).
    - \(n\) columns (one for each input variable).


**Notation and Interpretation**
- The Jacobian matrix is often represented as:

\[
\mathbf{J} = 
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
\]

- **Key insight:**
  - Each **row** focuses on a specific output \(f_i(\mathbf{x})\), showing how it changes with all input variables.
  - Each **column** focuses on a specific input variable \(x_j\), showing how all outputs \(f_1, f_2, \dots, f_m\) change with respect to \(x_j\).

##### 2. Two kinds of Jacobian Matrix 


**两种布局的区别**
| **布局形式**         | **分子布局 (Numerator-Layout)**                   | **分母布局 (Denominator-Layout)**               |
|--------------------|---------------------------------------------|-------------------------------------------|
| **定义方式**         | 输出变量的数量 × 输入变量的数量                     | 输入变量的数量 × 输出变量的数量                   |
| **Jacobian 矩阵维度** | \(\mathbb{R}^{N \times M}\)：矩阵是“高而瘦”的         | \(\mathbb{R}^{M \times N}\)：矩阵是“矮而胖”的      |
| **适用场景**         | 更常见于数学分析或机器学习中，尤其是梯度计算           | 常用于物理学或工程学中，尤其是优化问题             |
| **形象比喻**         | 一只**高而瘦的长颈鹿**                              | 一只**矮而胖的长颈鹿**                           |

 **公式对比**
假设函数为 \(\mathbf{y} = f(\mathbf{x})\)，其中：
- \(\mathbf{y} \in \mathbb{R}^N\) 是 \(N\) 维输出向量。
- \(\mathbf{x} \in \mathbb{R}^M\) 是 \(M\) 维输入向量。

*分子布局 (Numerator-Layout)*:
\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_M} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_M} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_N}{\partial x_1} & \frac{\partial y_N}{\partial x_2} & \cdots & \frac{\partial y_N}{\partial x_M}
\end{pmatrix}, \quad \text{维度：} N \times M
\]

- **行**：表示每个输出变量 \(y_i\) 对所有输入变量 \(x_j\) 的偏导数。
- **列**：表示每个输入变量 \(x_j\) 对所有输出变量 \(y_i\) 的影响。

*分母布局 (Denominator-Layout)*:
\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \cdots & \frac{\partial y_N}{\partial x_1} \\
\frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_N}{\partial x_2} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_M} & \frac{\partial y_2}{\partial x_M} & \cdots & \frac{\partial y_N}{\partial x_M}
\end{pmatrix}, \quad \text{维度：} M \times N
\]

- **行**：表示每个输入变量 \(x_j\) 对所有输出变量 \(y_i\) 的偏导数。
- **列**：表示每个输出变量 \(y_i\) 对所有输入变量 \(x_j\) 的影响。



---

 **举例说明**
假设我们有一个函数：
\[
\mathbf{y} = f(\mathbf{x}), \quad \mathbf{y} = 
\begin{pmatrix}
y_1 \\
y_2
\end{pmatrix}, \quad \mathbf{x} = 
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}.
\]

###### **分子布局 (Numerator-Layout)**:
- 输出维度 \(N = 2\)，输入维度 \(M = 3\)。
- Jacobian 矩阵为 \(2 \times 3\)：
\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \frac{\partial y_1}{\partial x_3} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \frac{\partial y_2}{\partial x_3}
\end{pmatrix}.
\]

###### **分母布局 (Denominator-Layout)**:
- 输出维度 \(N = 2\)，输入维度 \(M = 3\)。
- Jacobian 矩阵为 \(3 \times 2\)：
\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} \\
\frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} \\
\frac{\partial y_1}{\partial x_3} & \frac{\partial y_2}{\partial x_3}
\end{pmatrix}.
\]


**总结**
- **分子布局 (Numerator-Layout)** 更适合描述输出变量的变化，常用于梯度计算和机器学习。
- **分母布局 (Denominator-Layout)** 更适合描述输入变量的变化，常用于优化问题和工程应用。
- 两种布局只是表示方式不同，本质上提供的是相同的信息，具体选择取决于应用场景和习惯。

**Highlighted Notes**
- **Rows are about an entry of output.**
  - Each row corresponds to how one specific output component depends on all the input variables.
- **Columns are about an entry of input.**
  - Each column corresponds to how all output components depend on one specific input variable.

##### 3. 张量与输入输出
<img src="p7.png" width="70%" height="70%" >

###### 四维张量的形成
<img src="p2.png" width="90%" height="90%" >

###### **a. 向量输出的函数**
函数形式为：
\[
\mathbf{y} = f(\mathbf{x}),
\]
其中：
- **输入**：\(\mathbf{x} \in \mathbb{R}^M\)，是一个 \(M\) 维向量。
- **输出**：\(\mathbf{y} \in \mathbb{R}^N\)，是一个 \(N\) 维向量。

 **如何计算导数？**
- 每个输出变量 \(y_i\) 对每个输入变量 \(x_j\) 的偏导数 \(\frac{\partial y_i}{\partial x_j}\) 构成一个矩阵：
\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_M} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_N}{\partial x_1} & \cdots & \frac{\partial y_N}{\partial x_M}
\end{pmatrix}.
\]
- 这个矩阵表示 **Jacobian 矩阵**，维度为 \(N \times M\)。

 **总结：**
- 如果函数的输出是一个向量，导数将形成一个二维矩阵（Jacobian 矩阵）。


###### **b. 矩阵输出的函数**
函数形式为：
\[
\mathbf{Y} = f(\mathbf{X}),
\]
其中：
- **输入**：\(\mathbf{X} \in \mathbb{R}^{M \times L}\)，是一个 \(M \times L\) 的矩阵。
- **输出**：\(\mathbf{Y} \in \mathbb{R}^{N \times K}\)，是一个 \(N \times K\) 的矩阵。

#### **如何计算导数？**
- 每个输出元素 \(Y_{i,k}\) 对每个输入元素 \(X_{j,l}\) 的偏导数 \(\frac{\partial Y_{i,k}}{\partial X_{j,l}}\) 构成一个 **四维张量**：
\[
\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} \quad \text{是一个四维张量，维度为 } (N \times K \times M \times L).
\]

#### **总结：**
- 如果函数的输出是一个矩阵，导数将形成一个四维张量。
- 这种情况下，偏导数的维度比简单的向量输出更高，需要更多结构来表示。

---

###### **c. 幻灯片内容逐步解析**
#### **左侧：向量输出**
- 输入是一个 \(M\) 维向量，输出是一个 \(N\) 维向量。
- 通过计算偏导数 \(\frac{\partial y_i}{\partial x_j}\)，形成一个 \(N \times M\) 的矩阵（Jacobian 矩阵）。

#### **右侧：矩阵输出**
- 输入是一个 \(M \times L\) 的矩阵，输出是一个 \(N \times K\) 的矩阵。
- 通过计算偏导数 \(\frac{\partial Y_{i,k}}{\partial X_{j,l}}\)，形成一个 **四维张量**。

#### **关键点：**
- **Bag of Derivatives**：可以将这些偏导数看作是一个“导数集合”，具体的组织形式取决于输入和输出的维度。
- **四维张量**：当输入和输出都是矩阵时，导数的维度升高，形成一个 \(N \times K \times M \times L\) 的四维结构。


###### **d. 总结**
- **向量输出**的函数：导数形成一个二维矩阵（Jacobian 矩阵）。
- **矩阵输出**的函数：导数形成一个四维张量。
- 这种导数的组织方式在机器学习、优化以及深度学习中非常重要，尤其是在计算梯度时。

##### 4. Matrices and the Chain Rule
The slides explain **composite functions involving matrices** and how to apply the **chain rule** to such functions. Below is a detailed breakdown of the content from the slides.


**a. Composite Functions**
- A **composite function** takes a matrix as input, produces another matrix as output, and this output can be the input for another function.
- Mathematically, a composite function is written as:
  \[
  (f \circ g)(X) = f(g(X)),
  \]
  where \(X\) is a matrix, and \(f\) and \(g\) are matrix-valued functions.

**Example**
- If \(f(X) = AX\) and \(g(Y) = YB\), where \(A\) and \(B\) are matrices, then:
  \[
  (f \circ g)(X) = f(g(X)) = A(XB).
  \]
  This shows that the composite function applies \(g\) first (producing \(XB\)), and then applies \(f\) (resulting in \(A(XB)\)).

**Matrix Flow**
This process can be visualized as:
\[
X \to [g] \to Y \to [f] \to Z,
\]
where:
- \(X\) is the input matrix.
- \(g\) transforms \(X\) into \(Y\).
- \(f\) transforms \(Y\) into \(Z\).

**Chain Rule for Matrices**
The chain rule for composite functions involving matrices is:
\[
\frac{dZ}{dX} = \frac{dZ}{dY} \cdot \frac{dY}{dX}.
\]
- Here, **matrix multiplication order matters** because matrix multiplication is **not commutative** (i.e., \(AB \neq BA\) in general).
- The order of derivatives is crucial to ensure the correct computation of the result.
**Key Point**
- **Be careful of the order of operations** when applying the chain rule to matrices, as the non-commutativity of matrix multiplication can lead to errors.

---

**Example of the Chain Rule for Vector-Valued Functions**

This slide provides a concrete example of using the chain rule for a **composite function** involving vector-valued functions.

#### **Setup**
- Define a composite function \(h(t) = (f \circ g)(t) = f(g(t))\), where:
  - \(f : \mathbb{R}^2 \to \mathbb{R}\), \(f(x_1, x_2) = \exp(x_1 x_2)\),
  - \(g : \mathbb{R} \to \mathbb{R}^2\), \(g(t) = \begin{pmatrix} t \cos(t) \\ t \sin(t) \end{pmatrix}\).

#### **Notes**
1. The derivative of \(f\) with respect to its input \(\mathbf{x} = (x_1, x_2)\) is:
   \[
   \frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times 2}.
   \]
2. The derivative of \(g\) with respect to \(t\) is:
   \[
   \frac{\partial g}{\partial t} \in \mathbb{R}^{2 \times 1}.
   \]

#### **Using the Chain Rule**
To compute \(\frac{dh}{dt}\), we use the chain rule:
\[
\frac{dh}{dt} = \frac{\partial f}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{x}}{\partial t}.
\]

Breaking it down step by step:
1. Compute \(\frac{\partial f}{\partial \mathbf{x}}\):
   \[
   f(x_1, x_2) = \exp(x_1 x_2), \quad \frac{\partial f}{\partial x_1} = \exp(x_1 x_2) x_2, \quad \frac{\partial f}{\partial x_2} = \exp(x_1 x_2) x_1.
   \]
   Therefore:
   \[
   \frac{\partial f}{\partial \mathbf{x}} = \begin{pmatrix} \exp(x_1 x_2) x_2 & \exp(x_1 x_2) x_1 \end{pmatrix}.
   \]

2. Compute \(\frac{\partial \mathbf{x}}{\partial t}\):
   \[
   g(t) = \begin{pmatrix} t \cos(t) \\ t \sin(t) \end{pmatrix}, \quad
   \frac{\partial x_1}{\partial t} = \cos(t) - t \sin(t), \quad
   \frac{\partial x_2}{\partial t} = \sin(t) + t \cos(t).
   \]
   Therefore:
   \[
   \frac{\partial \mathbf{x}}{\partial t} = \begin{pmatrix} \cos(t) - t \sin(t) \\ \sin(t) + t \cos(t) \end{pmatrix}.
   \]

3. Multiply the results:
   \[
   \frac{dh}{dt} = \frac{\partial f}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{x}}{\partial t}.
   \]
   Substituting:
   \[
   \frac{dh}{dt} = \begin{pmatrix} \exp(x_1 x_2) x_2 & \exp(x_1 x_2) x_1 \end{pmatrix}
   \cdot
   \begin{pmatrix} \cos(t) - t \sin(t) \\ \sin(t) + t \cos(t) \end{pmatrix}.
   \]



## 3. Gradients of Matrices
<img src="p3.png" width="90%" height="90%" >


### **示例设定**
假设 \(\mathbf{R}\) 是一个 \(3 \times 3\) 的矩阵：
\[
\mathbf{R} = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix},
\]
则 \(\mathbf{K} = \mathbf{R}^\top \mathbf{R}\) 为：
\[
\mathbf{K} = \begin{bmatrix}
a^2 + d^2 + g^2 & ab + de + gh & ac + df + gi \\
ba + ed + hg & b^2 + e^2 + h^2 & bc + ef + hi \\
ca + fd + ig & cb + fe + ih & c^2 + f^2 + i^2
\end{bmatrix}.
\]
由于 \(\mathbf{K}\) 是对称矩阵（\(K_{pq} = K_{qp}\)），我们只需计算部分元素即可。


### **梯度计算**
#### **1. 当 \(p=1, q=1\)（对角线元素 \(K_{11}\)）**
\[
K_{11} = a^2 + d^2 + g^2.
\]
- **对 \(R_{i1}\) 求导（即 \(j=1\)，且 \(p=q=1\)）：**
  \[
  \frac{\partial K_{11}}{\partial R_{i1}} = 2R_{i1} \quad \Rightarrow \quad
  \begin{cases}
  \frac{\partial K_{11}}{\partial a} = 2a, \\
  \frac{\partial K_{11}}{\partial d} = 2d, \\
  \frac{\partial K_{11}}{\partial g} = 2g.
  \end{cases}
  \]
- **对其他列（如 \(j=2,3\)）：**
  \[
  \frac{\partial K_{11}}{\partial R_{ij}} = 0 \quad (\text{当 } j \neq 1).
  \]

#### **2. 当 \(p=1, q=2\)（非对角线元素 \(K_{12}\)）**
\[
K_{12} = ab + de + gh.
\]
- **对 \(R_{i1}\) 求导（\(j=p=1\)，且 \(p \neq q\)）：**
  \[
  \frac{\partial K_{12}}{\partial R_{i1}} = R_{i2} \quad \Rightarrow \quad
  \begin{cases}
  \frac{\partial K_{12}}{\partial a} = b, \\
  \frac{\partial K_{12}}{\partial d} = e, \\
  \frac{\partial K_{12}}{\partial g} = h.
  \end{cases}
  \]
- **对 \(R_{i2}\) 求导（\(j=q=2\)，且 \(p \neq q\)）：**
  \[
  \frac{\partial K_{12}}{\partial R_{i2}} = R_{i1} \quad \Rightarrow \quad
  \begin{cases}
  \frac{\partial K_{12}}{\partial b} = a, \\
  \frac{\partial K_{12}}{\partial e} = d, \\
  \frac{\partial K_{12}}{\partial h} = g.
  \end{cases}
  \]
- **对其他列（如 \(j=3\)）：**
  \[
  \frac{\partial K_{12}}{\partial R_{i3}} = 0.
  \]

#### **3. 当 \(p=2, q=3\)（非对角线元素 \(K_{23}\)）**
\[
K_{23} = bc + ef + hi.
\]
- **对 \(R_{i2}\) 求导（\(j=p=2\)，且 \(p \neq q\)）：**
  \[
  \frac{\partial K_{23}}{\partial R_{i2}} = R_{i3} \quad \Rightarrow \quad
  \begin{cases}
  \frac{\partial K_{23}}{\partial b} = c, \\
  \frac{\partial K_{23}}{\partial e} = f, \\
  \frac{\partial K_{23}}{\partial h} = i.
  \end{cases}
  \]
- **对 \(R_{i3}\) 求导（\(j=q=3\)，且 \(p \neq q\)）：**
  \[
  \frac{\partial K_{23}}{\partial R_{i3}} = R_{i2} \quad \Rightarrow \quad
  \begin{cases}
  \frac{\partial K_{23}}{\partial c} = b, \\
  \frac{\partial K_{23}}{\partial f} = e, \\
  \frac{\partial K_{23}}{\partial i} = h.
  \end{cases}
  \]
- **对其他列（如 \(j=1\)）：**
  \[
  \frac{\partial K_{23}}{\partial R_{i1}} = 0.
  \]


### **梯度张量表示**
梯度张量 \(\frac{\partial \mathbf{K}}{\partial \mathbf{R}} \in \mathbb{R}^{(3 \times 3) \times (3 \times 3)}\) 的每个元素对应 \(\frac{\partial K_{pq}}{\partial R_{ij}}\)。部分结果如下：

| \(K_{pq}\) | \(\frac{\partial K_{pq}}{\partial \mathbf{R}}\) 的梯度矩阵（\(3 \times 3\)） |
|----------|------------------------------------------------|
| \(K_{11}\) | \(\begin{bmatrix} 2a & 0 & 0 \\ 2d & 0 & 0 \\ 2g & 0 & 0 \end{bmatrix}\) |
| \(K_{12}\) | \(\begin{bmatrix} b & a & 0 \\ e & d & 0 \\ h & g & 0 \end{bmatrix}\) |
| \(K_{23}\) | \(\begin{bmatrix} 0 & c & b \\ 0 & f & e \\ 0 & i & h \end{bmatrix}\) |
| \(K_{33}\) | \(\begin{bmatrix} 0 & 0 & 2c \\ 0 & 0 & 2f \\ 0 & 0 & 2i \end{bmatrix}\) |


### **总结**
1. **对角线元素 \(K_{pp}\)：**
   - 梯度仅在第 \(p\) 列非零，且为对应元素的 2 倍。
   - 例如，\(K_{33}\) 的梯度矩阵中，只有第 3 列有非零值。

2. **非对角线元素 \(K_{pq}\)：**
   - 梯度矩阵在第 \(p\) 列和第 \(q\) 列非零。
   - 第 \(p\) 列的导数是 \(\mathbf{R}\) 的第 \(q\) 列元素，第 \(q\) 列的导数是 \(\mathbf{R}\) 的第 \(p\) 列元素。

3. **四维张量结构：**
   - 每个 \(K_{pq}\) 对应一个 \(3 \times 3\) 的梯度矩阵。
   - 总维度为 \((3 \times 3) \times (3 \times 3)\)，共 81 个元素。

如果有其他问题，欢迎继续提问！

## 4. Backpropagation and Automatic Differentiation
<img src="p5.png" width="90%" height="90%" >
<img src="p8.png" width="90%" height="90%" >

### **4.1 典型分类器（Classifier）**
<img src="p6.png" width="90%" height="90%" >

 **1. 输入/输出对 \((x^{(i)}, y^{(i)})\)** 
- **输入 \((x^{(i)})\)**：表示第 \(i\) 个样本的特征向量，例如 \([x_1, x_2, ..., x_n]\)。每个 \(x_j\) 是样本的一个特征。
  - 举例：在图像分类中，\(x_j\) 可以是像素值。
- **输出 \((y^{(i)})\)**：表示第 \(i\) 个样本的真实类别标签。
  - 举例：在二分类问题中，\(y^{(i)}\) 可能是 \(0\)（负类）或 \(1\)（正类）。



 **2. 特征表示（Feature Representation）**
- **定义**：对输入数据进行特征化，形成一个特征向量 \([x_1, x_2, ..., x_n]\)，用于表示样本 \(x^{(i)}\) 的属性。
- **形式**：
  - \(x_j\)：直接使用的第 \(j\) 个特征值。
  - \(f_j(x)\)：有时会用函数 \(f_j(x)\) 对特征进行变换（如归一化、标准化或特征提取）。


 **3. 分类函数（Classification Function）**
- **定义**：根据输入特征 \(x^{(i)}\)，计算预测类别 \(\hat{y}\)。
- **形式**：
  - \(p(y|x)\)：表示给定输入 \(x\)，预测类别 \(y\) 的概率。
  - 常用函数：
    - **Sigmoid 函数**：用于二分类问题，将输出映射到 \([0, 1]\) 的概率范围。
    - **Softmax 函数**：用于多分类问题，输出每个类别的概率分布。


 **4. 目标函数（Objective Function）**
- **定义**：用于衡量模型预测值 \(\hat{y}\) 与真实值 \(y^{(i)}\) 的差距，即**损失函数（Loss Function）**。
- **常用目标函数**：
  - **交叉熵损失（Cross-Entropy Loss）**：常用于分类任务，特别是概率输出的模型。
    - 二分类交叉熵公式：  
      \[
      \text{Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
      \]

 **5. 优化算法（Optimization Algorithm）**
- **定义**：通过优化算法最小化目标函数，从而找到最佳模型参数。
- **常用方法**：
  - **梯度下降（Gradient Descent）**：通过计算损失函数相对于参数的梯度，沿反梯度方向调整参数，逐步逼近最优解。
  - 更新公式：
    \[
    w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla_w \text{Loss}
    \]
    - \(w\)：模型参数
    - \(\eta\)：学习率（步长）

### **4.2 A Recipe for Machine Learning**
<img src="p9.png" width="90%" height="90%" >



#### **1. Given Training Data**
\[
\{x_i, y_i\}_{i=1}^N
\]
- **Input:** A dataset consisting of \( N \) samples.
  - \( x_i \): The input features for the \( i \)-th sample.
  - \( y_i \): The corresponding label or target value for the \( i \)-th sample.

The goal is to use this dataset to train a model that can predict \( y_i \) given \( x_i \).

#### **2. Choose Components**
Two key components must be defined for the machine learning model:

##### **a. Decision Function**
\[
\hat{y} = f_\theta(x_i)
\]
- \( f_\theta(x_i) \): The model (or hypothesis function) parameterized by \( \theta \), which maps input \( x_i \) to a predicted output \( \hat{y} \).
- Example:
  - For a linear model: \( f_\theta(x_i) = \theta^T x_i \).
  - For a neural network: \( f_\theta(x_i) \) could involve multiple layers and non-linear transformations.

##### **b. Loss Function**
\[
\ell(\hat{y}, y_i) \in \mathbb{R}
\]
- The **loss function** measures the difference between the predicted value \( \hat{y} \) and the true value \( y_i \).
- Example:
  - For regression: Mean Squared Error (MSE): \( \ell(\hat{y}, y_i) = (\hat{y} - y_i)^2 \).
  - For classification: Cross-Entropy Loss.

#### **3. Define the Goal**
\[
\theta^* = \arg \min_\theta \sum_{i=1}^N \ell(f_\theta(x_i), y_i)
\]
- **Objective:** Minimize the total loss across all training samples by finding the optimal parameters \( \theta^* \).
- The goal is to adjust \( \theta \) such that the predictions \( f_\theta(x_i) \) are as close as possible to the true labels \( y_i \).

#### **4. Train with Stochastic Gradient Descent (SGD)**
\[
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \ell(f_\theta(x_i), y_i)
\]
- **Gradient Descent:** Iteratively update the parameters \( \theta \) by moving in the direction opposite to the gradient of the loss function.
  - \( \nabla_\theta \ell(f_\theta(x_i), y_i) \): The gradient of the loss function with respect to \( \theta \).
  - \( \eta \): The learning rate, which controls the step size of each update.
- **Stochastic Gradient Descent (SGD):** Updates \( \theta \) using one sample (or a small batch) at a time, rather than the entire dataset.

### 4.3 Linear regression 
<img src="p10.png" width="90%" height="90%" >

<font color='blue'>It does not directly handle categorical outputs, which are required for classification tasks.</font>
In machine learning, \(\sigma(a)\) is often used to represent an **activation function** . An activation function is a mathematical transformation applied to the output of a model (or neuron in neural networks). Its purpose is to introduce **non-linearity** to the model.

- If \(\sigma(a)\) is a non-linear function (e.g., sigmoid, ReLU, tanh), the model becomes **non-linear** because the output is no longer a simple linear combination of the inputs.
- If \(\sigma(a) = a\), this means that the activation function is the **identity function**, i.e., it doesn't change the value of \(a\). This keeps the model **linear**.

If \(\sigma(a)\) is a non-linear function, such as:
- **Sigmoid**: \(\sigma(a) = \frac{1}{1 + e^{-a}}\)
- **ReLU**: \(\sigma(a) = \max(0, a)\)
- **Tanh**: \(\sigma(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}\)
### 4.4 Logistic Regression
<img src="p11.png" width="90%" height="90%" >


### 4.5 Backpropagation
<img src="p12.png" width="90%" height="90%" >

This image explains the **Backpropagation Algorithm**, which is primarily used to calculate gradients for optimizing model parameters (e.g., weights \(\theta\)). It is a crucial part of deep learning, especially when training neural networks. Below is a detailed explanation of the content in the image:

### **1. Structure of the Diagram**
- **Top-left part**: Shows a simple neural network structure where the output is \(y\), and the inputs are \(x_1, x_2, x_3, \dots, x_M\). Each input has a corresponding weight \(\theta_1, \theta_2, \theta_3, \dots, \theta_M\).
- **Bottom-right part**: Divided into two sections:
  - **Forward**: Computes the output \(y\) and the loss function \(J\).
  - **Backward**: Computes gradients \(\frac{\partial J}{\partial \theta_j}\) and \(\frac{\partial J}{\partial x_j}\), which are used to update the parameters.

### **2. Forward Propagation**
In forward propagation, the following calculations are performed:

1. **Activation Value \(a\)**:
   \[
   a = \sum_{j=0}^D \theta_j x_j
   \]
   This is the weighted sum of the input features \(x_j\) and their corresponding weights \(\theta_j\).

2. **Output \(y\)**:
   Using the sigmoid activation function:
   \[
   y = \frac{1}{1 + \exp(-a)}
   \]
   The sigmoid function maps \(a\) to the range \([0, 1]\), which is commonly used for binary classification tasks to predict probabilities.

3. **Loss Function \(J\)**:
   Using the cross-entropy loss function:
   \[
   J = y^* \log(y) + (1 - y^*) \log(1 - y)
   \]
   - \(y^*\) is the true label (target value).
   - \(y\) is the model's predicted output.
   - This function measures the difference between the predicted value \(y\) and the true label \(y^*\), serving as the objective to minimize during optimization.

### **3. Backward Propagation**
The purpose of backpropagation is to compute the gradients of the loss function \(J\) with respect to the parameters (e.g., \(\theta_j\) and \(x_j\)) for updating the model. The key steps are as follows:
先通过通过链式法则（Chain Rule），计算损失函数对每个权重的偏导数（梯度）。  
- 首先计算损失对输出的梯度。
- 然后逐层向后传播，计算每一层的梯度。
  
再使用梯度下降法，根据计算出的梯度调整权重和偏置。  
公式：
\[
W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial \text{Loss}}{\partial W}
\]  
其中：
- \(W_{\text{new}}\)：更新后的权重
- \(\eta\)：学习率（控制更新步长）
- \(\frac{\partial \text{Loss}}{\partial W}\)：损失对权重的梯度


**(1) Gradient of Loss with Respect to Output \(\frac{\partial J}{\partial y}\):**
\[
\frac{\partial J}{\partial y} = \frac{y^*}{y} + \frac{1 - y^*}{y - 1}
\]
This step computes how the loss function \(J\) changes with respect to the model's output \(y\).

**(2) Gradient of Output with Respect to Activation \(\frac{\partial y}{\partial a}\):**
The derivative of the sigmoid function is:
\[
\frac{\partial y}{\partial a} = \frac{\exp(-a)}{(\exp(-a) + 1)^2}
\]
This step computes how the model's output \(y\) changes with respect to the activation value \(a\).

**(3) Gradient of Loss with Respect to Activation \(\frac{\partial J}{\partial a}\):**
Using the chain rule:
\[
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial y} \cdot \frac{\partial y}{\partial a}
\]
Combining the results from the previous two steps gives the gradient of the loss function with respect to the activation value \(a\).

**(4) Gradient of Loss with Respect to Weights \(\frac{\partial J}{\partial \theta_j}\):**
Using the chain rule:
\[
\frac{\partial J}{\partial \theta_j} = \frac{\partial J}{\partial a} \cdot \frac{\partial a}{\partial \theta_j}
\]
Since \(a = \sum_{j=0}^D \theta_j x_j\), we have:
\[
\frac{\partial a}{\partial \theta_j} = x_j
\]
Thus:
\[
\frac{\partial J}{\partial \theta_j} = \frac{\partial J}{\partial a} \cdot x_j
\]

**(5) Gradient of Loss with Respect to Inputs \(\frac{\partial J}{\partial x_j}\):**
Similarly, using the chain rule:
\[
\frac{\partial J}{\partial x_j} = \frac{\partial J}{\partial a} \cdot \frac{\partial a}{\partial x_j}
\]
Since \(a = \sum_{j=0}^D \theta_j x_j\), we have:
\[
\frac{\partial a}{\partial x_j} = \theta_j
\]
Thus:
\[
\frac{\partial J}{\partial x_j} = \frac{\partial J}{\partial a} \cdot \theta_j
\]

### **4. Summary**
This diagram illustrates the complete process of backpropagation:
- **Forward Propagation**: Computes the output \(y\) and the loss \(J\).
- **Backward Propagation**: Uses the chain rule to compute gradients, including \(\frac{\partial J}{\partial \theta_j}\) and \(\frac{\partial J}{\partial x_j}\).

These gradients are then used to update the model parameters (e.g., weights \(\theta_j\)) to minimize the loss function \(J\). This is the core of training a neural network.

## 错题
### 单变量求导
#### Problem 5: Rational and Exponential Functions

Find the derivative of:
\[
f(x) = \frac{e^{x^2 + 1}}{\sqrt{1 + x^4}}
\]

---

###### Solution:

Using the quotient rule:
\[
f'(x) = \frac{\frac{d}{dx}[e^{x^2 + 1}] \cdot \sqrt{1 + x^4} - e^{x^2 + 1} \cdot \frac{d}{dx}[\sqrt{1 + x^4}]}{(1 + x^4)}
\]


Substitute the derivatives into the quotient rule:
\[
f'(x) = \frac{(2x e^{x^2 + 1}) \sqrt{1 + x^4} - e^{x^2 + 1} \cdot \frac{4x^3}{2\sqrt{1 + x^4}}}{1 + x^4}
\]

Simplify:
\[
f'(x) = \frac{2x e^{x^2 + 1} \sqrt{1 + x^4} - 2x^3 e^{x^2 + 1} / \sqrt{1 + x^4}}{1 + x^4}
\]

Factor out \(2x e^{x^2 + 1}\):
\[
f'(x) = \frac{2x e^{x^2 + 1} \left(\sqrt{1 + x^4} - \frac{x^2}{\sqrt{1 + x^4}}\right)}{1 + x^4}
\]

Simplify further if needed:
\[
f'(x) = \frac{2x e^{x^2 + 1} \cdot \frac{1 + x^4 - x^2}{\sqrt{1 + x^4}}}{1 + x^4}
\]
\[
f'(x) = \frac{2x e^{x^2 + 1} \cdot \frac{(1 + x^2)(1 - x^2)}{\sqrt{1 + x^4}}}{1 + x^4}
\]
#### 多变量
##### Example: Partial Derivatives of a Function

Given the function:

\[
f(x, y) = (x + 2y^3)^2
\]



Final Results:

\[
\frac{\partial f(x, y)}{\partial x} = 2(x + 2y^3)
\]

\[
\frac{\partial f(x, y)}{\partial y} = 12(x + 2y^3)y^2
\]

---
在例子中，函数 \(f(\mathbf{x})\) 被定义为：

\[
f(\mathbf{x}) = 
\begin{pmatrix}
x_1^2 + x_2 \\
x_1 x_2 \\
\sin(x_1)
\end{pmatrix}, \quad \mathbf{x} = 
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}
\]

我们需要对 \(f(\mathbf{x})\) 的每个分量分别对 \(x_2\) 求偏导。现在重点分析第三个分量 \(\sin(x_1)\) 对 \(x_2\) 的偏导数。



### **第三个分量 \(\sin(x_1)\) 的偏导分析**
- \(\sin(x_1)\) 是一个关于 \(x_1\) 的函数，它完全独立于 \(x_2\)，即它不包含 \(x_2\)。
- 因为 \(\sin(x_1)\) 中没有 \(x_2\)，所以对 \(x_2\) 求偏导时，结果为 0。

**数学表达：**
\[
\frac{\partial}{\partial x_2} \sin(x_1) = 0
\]


### **为什么偏导为 0？**
偏导数表示一个变量的变化如何影响函数值。在 \(\sin(x_1)\) 中，\(x_2\) 不会对函数值产生任何影响，因为它根本没有出现在函数中。因此，对 \(x_2\) 求偏导的结果就是 0。


### **总结**
在例子中，函数的第三个分量是 \(\sin(x_1)\)，它只与 \(x_1\) 有关，与 \(x_2\) 无关。所以对 \(x_2\) 求偏导时，结果为 0。这就是为什么雅可比矩阵的第三行第二列是 0 的原因。

---
<img src="p4.png" width="60%" height="60%" >




