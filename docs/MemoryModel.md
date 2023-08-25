# Exocortex Memory Model

The curves for short-term and long-term memories are exponential and logarithmic respectively, normalizing and adapting to both the oldest long-term memory and the short-term memory duration specified by the user.

## Decay Rate and Threshold functions 

**Set by the end user**:
- Short-term memory duration 
- Long-term decay threshold
- The rest is calculated based on these two values

**Short-term Memory Decay**:
- **Function**: Exponential decay
- **Equation**: \( short_term_decay(t)=exp(−short_term_decay_rate×t) \)
- **Rationale**: Captures the rapid fading of recent memories, consistent with how short-term human memories fade quickly.
   
**Long-term Memory Decay**:
- **Function**: Reversed logarithmic decay
- **Rationale**: Mimics human memory patterns where memories fade slowly over time but never completely disappear. The initial rapid decay slows down, representing the long-lasting nature of certain memories.
   
**Short-term Decay Threshold**:

- **Value**: Computed exponentially using the long term duration.
- **Function**: Exponential
- **Role**: Represents the point at which short-term memories start transitioning to long-term memories. As more long-term memories accumulate, the threshold decreases, leading short-term memories to decay faster.
- **Challenge & Solution**: 
    - It was observed after computing this based on long-term duration that the max possible threshold (1) was being approached exponentially with age, leading to rapid memory loss around age 45.
    - The issue stems from how we dynamically adjust the short-term decay threshold based on the long-term duration. As the long-term duration increases (i.e., as someone gets older), the short-term decay threshold increases, which leads to a noticeable decay of short-term memory.
    - The exponential function is chosen because it is the inverse of the logarithmic function used to calculate the long-term memory decay curve. This means that the difference between the short-term decay threshold and the long-term decay threshold is approached logistically, which results in a flattening and decline in long-term memory strength.

 **Long-term Decay Threshold**:
- **Value**: Constant at 0.01, can be changed by user.
- **Function**: Exponential
- **Role**: Represents the minimum strength a long-term memory can have. It's the point below which a memory is considered lost or inaccessible.
- **Rationale**: Mimics the idea that some human memories, no matter how old, never completely vanish but can become very hard to access.
- **Challenge & Solution**:
  - By making the short-term decay threshold climb exponentially with age (countering the age 45 memory loss), we noted the functional inverse was happening - the difference between the short term decay threshold and the long-term decay threshold was being approached logistically.
  - This resulted in a flattening and decline in long-term memory strength. This was rectified by applying an exponential function to counter it.
   
<img src="adjusted_memory_decay_Infancy_to_Early_Childhood.png">
<br/>
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
    <img src="adjusted_memory_decay_Childhood_to_Adolescence.png">
    <img src="adjusted_memory_decay_Young_Adulthood.png" >
    <img src="adjusted_memory_decay_Middle_Age.png">
</div>
<br/>
<img src="adjusted_memory_decay_Elderly.png">
