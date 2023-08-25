# Exocortex Memory Model

## Decay Rate and Threshold functions 

**Set by the end user**:
- Short-term memory duration 
- Long-term decay threshold
- The rest is calculated for you

**Short-term Memory Decay**:
- **Function**: Exponential decay
- **Equation**: \( e^{-\text{{short-term decay rate}} \times t} \)
- **Rationale**: Captures the rapid fading of recent memories, consistent with how short-term human memories fade quickly.
   
**Long-term Memory Decay**:
- **Function**: Reversed logarithmic decay
- **Equation**: \( a - b \times \log(c \times t + 1) \)
- **Rationale**: Mimics human memory patterns where memories fade slowly over time but never completely disappear. The initial rapid decay slows down, representing the long-lasting nature of certain memories.
   
**Short-term Decay Threshold**:
- **Function**: Exponential
- **Equation**: \( \frac{1}{{0.00005 \times \text{{long-term duration}} + 1}} \)
- **Role**: Represents the point at which short-term memories start transitioning to long-term memories. As more long-term memories accumulate, the threshold decreases, leading short-term memories to decay faster.
- **Challenge & Solution**: It was observed after computing this based on long-term duration that the max possible threshold (1) was being approached exponentially with age, leading to rapid memory loss around age 45. An exponential function was chosen to counteract this effect.
   
 **Long-term Decay Threshold**:
- **Value**: Constant at 0.01, can be changed by user.
- **Role**: Represents the minimum strength a long-term memory can have. It's the point below which a memory is considered lost or inaccessible.
- **Rationale**: Mimics the idea that some human memories, no matter how old, never completely vanish but can become very hard to access.
- **Challenge & Solution**: By making the short-term decay threshold climb exponentially with age (countering the age 45 memory loss), we noted the functional inverse was happening - the difference between the short term decay threshold and the long-term decay threshold was being approached logistically. This resulted in a flattening and decline in long-term memory strength. This was rectified by applying an exponential function to counter it.
   
