# Introduction to Attention

### 1. What is Attention Mechanism?  
- **Introduction**:  
  - When processing sequential data (e.g., text), how can the model focus on the important parts based on the context?  
  - For example, in the sentence "I like eating apples," the model should focus on "like" and "eating" when predicting "apples."  
- **Core Idea of Attention Mechanism**:  
  - Attention mechanism calculates **the relationships between each position in the input sequence** to decide how to allocate attention.  

### 2. Mathematical Definition of Attention Mechanism  
- **Input**:  
  - The sequence is represented as a matrix $$X$$, where there are $$n$$ words, and each word is represented as a $$d$$-dimensional vector:  
```math
X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \quad \text{where } x_i \in \mathbb{R}^d
```

- **Core Computation**:  
  - For each input $$x_i$$, the attention mechanism calculates its relationship with other inputs $$x_j$$.  
  - Three matrices are used: **Query (Q)**, **Key (K)**, **Value (V)**.  
    $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$  
    - $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$$ are learnable weight matrices.  
    - $$d_k$$ is the dimensionality of the attention space.  

### 3. Attention Score Calculation  
- **Similarity Calculation**: Use dot product to measure the similarity between $$Q$$ and $$K$$.  
  $$\text{Score}(Q, K) = Q \cdot K^\top$$  
- **Normalization**: Use the Softmax function to normalize attention scores:  
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V$$  
  - $$\sqrt{d_k}$$: Scaling factor to prevent large dot product values from causing instability in gradients.  
  - Softmax function definition:  
    $$\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^n \exp(z_j)}$$  

### 4. Computation Workflow of Self-Attention  
- **Input Sequence**:  
  - Assume the input sequence is $$X = [x_1, x_2, x_3]$$, where each $$x_i$$ is a 2-dimensional vector.  
  - Example matrix:  
```math
    X = \begin{bmatrix}  
    1 & 0 \\  
    0 & 1 \\  
    1 & 1  
    \end{bmatrix}
``` 
- **Weight Matrices**:


  - Assume $$W_Q, W_K, W_V$$ are:
```math
    W_Q = \begin{bmatrix}  
    1 & 0 \\  
    0 & 1  
    \end{bmatrix}, \quad  
    W_K = \begin{bmatrix}  
    1 & 1 \\  
    0 & 1  
    \end{bmatrix}, \quad  
    W_V = \begin{bmatrix}  
    1 & 0 \\  
    0 & 1  
    \end{bmatrix}
```

- **Calculate $$Q, K, V$$**:
```math
Q = XW_Q = \begin{bmatrix}  
  1 & 0 \\  
  0 & 1 \\  
  1 & 1  
  \end{bmatrix} \cdot  
  \begin{bmatrix}  
  1 & 0 \\  
  0 & 1  
  \end{bmatrix} =  
  \begin{bmatrix}  
  1 & 0 \\  
  0 & 1 \\  
  1 & 1  
  \end{bmatrix}
``` 
  Similarly, calculate $$K$$ and $$V$$:  
```math

  K = XW_K = \begin{bmatrix}  
  1 & 1 \\  
  0 & 1 \\  
  1 & 2  
  \end{bmatrix}, \quad  
  V = XW_V = \begin{bmatrix}  
  1 & 0 \\  
  0 & 1 \\  
  1 & 1  
  \end{bmatrix}
```

- **Calculate Attention Scores**:
```math
  \text{Score}(Q, K) = Q \cdot K^\top = \begin{bmatrix}  
  1 & 0 \\  
  0 & 1 \\  
  1 & 1  
  \end{bmatrix} \cdot  
  \begin{bmatrix}  
  1 & 0 & 1 \\  
  1 & 1 & 2  
  \end{bmatrix}^\top =  
  \begin{bmatrix}  
  1 & 1 & 3 \\  
  1 & 1 & 2 \\  
  2 & 2 & 5  
  \end{bmatrix}
```

- **Normalize Attention Scores**:  
  Use Softmax normalization:  
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{\text{Score}(Q, K)}{\sqrt{d_k}}\right) \cdot V$$  
