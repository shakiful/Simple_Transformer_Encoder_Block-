# Simple_Transformer_Encoder_Block
# Homework 5 
**Course:** CS5710 – Machine Learning  
**Student Name:** Md Shakiful Islam Khan  
**Student ID:** 700778823  

## Implementation Details (Part B)

### Q1: Scaled Dot-Product Attention

I implemented the scaled dot-product attention mechanism from scratch using **NumPy**

* **Function:** `scaled_dot_product_attention(Q, K, V)`
* **Logic:**
    1.  Calculates dot products between Query ($Q$) and Key ($K$) matrices.
    2.  Scales the scores by $\sqrt{d_k}$ to prevent gradient instability.
    3.  Applies Softmax to normalize scores into probabilities.
    4.  Multiplies weights by Value ($V$) matrices to get the context vector.
 
### Q2: Transformer Encoder Block

I implemented a simplified Transformer Encoder block using **PyTorch**

1. **Multi-Head Attention:** Uses `nn.MultiheadAttention` with $d_{model}=128$ and $h=8$ heads.
2. **Feed-Forward Network:** A sequential network with two linear layers and a ReLU activation.
3. **Add & Norm:** Implements residual connections and Layer Normalization after both sub-layers.
4. **Verification:** The script includes a test case verifying the output shape is `(32, 10, 128)` for a batch of 32 sentences with 10 tokens each
