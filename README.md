# OwlCore.AI.Exocortex [![Version](https://img.shields.io/nuget/v/OwlCore.AI.Exocortex.svg)](https://www.nuget.org/packages/OwlCore.AI.Exocortex)

Inspired by the paper "Generative Agents: Interactive Simulacra of Human Behavior"[^1]

The creation of an Exocortex is part of a larger exploration into the "Symbiosis of Artificial and Human Intelligence"[^3].

> **Important**
> 
> While the research remains sound on paper, the working prototype is incomplete.
> 
> This project is on ice until I can dedicate more time to building with LLMs. 

> **Note**  
> Our finished research on memory curves now available in [docs/MemoryModel.md](https://github.com/Arlodotexe/OwlCore.AI.Exocortex/blob/main/docs/MemoryModel.md)

## **An Echo of Experiences**
**Definition**: The Exocortex is a **remembrance agent**, a generative agent designed to simulate the human brain's memory recall and consolidation processes.

The Exocortex operates on a "rolling context" mechanism, ensuring that the most recent and relevant memories are prioritized, mimicking the human brain's ability to keep track of an ongoing conversation by constantly updating its understanding based on new information.

**Privacy**: Users can swap the AI model and can create specialized exocortices for different domains.

### Memory Architecture
This architecture was originally inspired by "Generative Agents: Interactive Simulacra of Human Behavior", and has some significant modifications that cater to the nuances of human memory. 

For more details, see [docs/MemoryModel.md](https://github.com/Arlodotexe/OwlCore.AI.Exocortex/blob/main/docs/MemoryModel.md).

### Memory Weighting Formula

Memories in the Exocortex decay over time, reflecting the human tendency to recall recent memories more vividly than older ones. This decay is modeled using an intricate balance between duration, decay rate, and decay thresholds for both short-term and long-term memory.

The weighting formula prioritizes both the recency and relevance of memories, ensuring a balance between recalling recent experiences and older but more relevant memories.

The final weight of a memory is determined by factors such as relevance, recency, and type. By adjusting the weight using an inverse of the recency score, the Exocortex ensures that while recent memories are naturally prioritized, older but relevant memories aren't overshadowed.

For more details, see [docs/MemoryModel.md](https://github.com/Arlodotexe/OwlCore.AI.Exocortex/blob/main/docs/MemoryModel.md).

### Clustering, Consolidation and Recollections

- **Recollections as Short-term Memories**: By recalling a cluster of memories, the system also remembers the act of recalling within the context of the short-term memories. Recollections in the Exocortex are used for transferring memories from long-term into short-term and for building compact, dense, broad context long-term memories that are highly suitable for recollection.
  
- **Summarizing the Context**: Recollections are used to summarize memories that are relevant to the short-term memories with respect to a new input, storing that as a new recollection memory. These are more likely to be retrieved from long-term memory, provided as past context for a future conversation.
  - The summarization mechanism has since been briefly explored and confirmed in an academic setting. See "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models."[^4]. This does not touch on clustering, memory curves or other areas that still need formal research.

For more details, see [docs/MemoryModel.md](https://github.com/Arlodotexe/OwlCore.AI.Exocortex/blob/main/docs/MemoryModel.md).

### Memory Clustering

1. **Formation of Clusters**: When a new memory or prompt is introduced, the Exocortex identifies memories that are similar to the short-term memories, including the new prompt. These memories, though rooted in the same context, might differ slightly from each other.
2. **Memory Summaries**: For each memory cluster, a representative summary is created. These summaries are reflections of the memories within the cluster, rooted in the same context of the prompt but augmented with slight variations based on the memories within the cluster.
3. **Recollections as Short-term Memories**: An interesting aspect of the Exocortex's memory model is that recollections themselves are treated as short-term memories. This means that when the AI recalls a memory, it also remembers the act of recalling. This layered approach ensures that the AI not only recalls past memories but also continually reframes them in light of new experiences. 

For more details, see [docs/MemoryModel.md](https://github.com/Arlodotexe/OwlCore.AI.Exocortex/blob/main/docs/MemoryModel.md).

## Install

Published releases are available on [NuGet](https://www.nuget.org/packages/OwlCore.AI.Exocortex). To install, run the following command in the [Package Manager Console](https://docs.nuget.org/docs/start-here/using-the-package-manager-console).

    PM> Install-Package OwlCore.AI.Exocortex
    
Or using [dotnet](https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet)

    > dotnet add package OwlCore.AI.Exocortex

## Usage
TODO

```cs
var test = new Thing();
```

## Financing

We accept donations [here](https://github.com/sponsors/Arlodotexe) and [here](https://www.patreon.com/arlodotexe), and we do not have any active bug bounties.

## Versioning

Version numbering follows the Semantic versioning approach. However, if the major version is `0`, the code is considered alpha and breaking changes may occur as a minor update.

## License

All OwlCore code is licensed under the MIT License. OwlCore is licensed under the MIT License. See the [LICENSE](./src/LICENSE.txt) file for more details.

[^1]: Joon Sung Park and Joseph C. O'Brien and Carrie J. Cai and Meredith Ringel Morris and Percy Liang and Michael S. Bernstein (2023). Generative Agents: Interactive Simulacra of Human Behavior. arXiv preprint [arXiv:2304.03442](https://arxiv.org/abs/2304.03442).
[^2]: Vaishnavi Himakunthala and Andy Ouyang and Daniel Rose and Ryan He and Alex Mei and Yujie Lu and Chinmay Sonar and Michael Saxon and William Yang Wang (2023). Let’s Think Frame by Frame: Evaluating Video Chain of Thought with
Video Infilling and Prediction. arXiv preprint [arXiv:2305.13903](https://arxiv.org/abs/2305.13903).
[^3]: Arlo Godfrey: Exploring Symbiosis of Artificial and Human Intelligence. [brain-dump/6](https://github.com/Arlodotexe/brain-dump/issues/6)
[^4]: Qingyue Wang, Liang Ding, Yanan Cao, Zhiliang Tian, Shi Wang, Dacheng Tao, Li Guo (2023). "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models." arXiv preprint [arXiv:2308.15022](https://arxiv.org/abs/2308.15022).


