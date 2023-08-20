# OwlCore.AI.Exocortex [![Version](https://img.shields.io/nuget/v/OwlCore.AI.Exocortex.svg)](https://www.nuget.org/packages/OwlCore.AI.Exocortex)
The creation of an Exocortex is part of a larger exploration into the "Symbiosis of Artificial and Human Intelligence"[^3].

## **An Echo of Experiences**
**Definition**: The Exocortex is a **remembrance agent**, a generative agent specialized in remembering a narrative of observed events.

**Vision**: A timestamped and vectorized timeline of experiences and notes. Key to this vision is the concept of the memory stream, a dynamic entity that evolves with new experiences. It's designed to provide a sense of continuity, using careful retrieval of related memories to provide an ongoing context for new experiences.

**Enhancing communication**: The immediate application and likely testing grounds will be to create a timeline of real events and notes in a specific domain to create a basic prototype, with the aim to identify and eliminate communication bottlenecks. Through this, constructs can collaborate or exchange knowledge and experiences autonomously.

**Privacy**: First priority, never to be undermined. All computation should be performed on-device. The exact AI model used should be swappable by the end user. Users should be able to create separate exocortexes for work and personal life, with the option to combine them together.

### Memory Stream Architecture
The memory stream is highly inspired by that in "Generative Agents: Interactive Simulacra of Human Behavior", but with some small changes and big additions. 

- **Initial Memory Values**: 
    - **Recency**: Each new memory is timestamped upon creation. Recency decays over time, using an exponential decay function (e.g., decay factor of 0.995) based on the number of hours since the memory was last retrieved.
    - **Importance**: Assigned an initial score when created, indicating the poignancy of the memory on a scale from 1 (mundane) to 10 (extremely poignant).
    - **Relevance**: Initialized based on the context in which the memory was created, and updated based on the similarity between the memory’s embedding vector and the current context’s embedding vector.

- **Final Retrieval Score**: The retrieval function scores all memories as a weighted combination of recency, importance, and relevance. Scores are normalized to the range of [0, 1] using min-max scaling. In the current implementation, all weights (𝛼) are set to 1. The top-ranked memories that fit within the language model’s context window are included in the prompt.

- **Continuity Across Memories**: As new events or memory entries occur, the system retrieves past memories based on factors like time, relevance, and importance. This provides context for the new memory entry. Objective observations become personalized to past experiences, especially the immediate past.

- **Meta-Observations**: Removes the need to include the full memory transcript in the context window. 
  - By treating memory recollection as a new observation, it can add new context to old memories without overwriting them, and it boosts the odds of being recalled again in the near future, mimicking an organic working memory.
  - Doing this also changes how things are remembered through the lens of the active context.
  - This is reminiscent of how human memory works: recalling a memory can change how it is remembered, and the act of remembering can itself become a new memory.

### Memory Consolidation (SOM Sleep)
A Self-Organizing Map (SOM) is a type of artificial neural network that is trained using unsupervised learning.

- **Mimicking Human Sleep Cycles**: During 'SOM Sleep', the Exocortex would use SOMs to reorganize and consolidate memories, akin to how human brains are believed to consolidate memories during sleep. This process could involve strengthening important connections between memories and weakening or pruning less important ones.

- **Contextual Integration**: During SOM Sleep, the Exocortex could use SOMs to integrate recent memories with old ones, updating and recontextualizing short-term memories into long-term memories based on new information and reflections.

- **Memory Pruning and Cleanup**: SOM Sleep could also involve cleaning up the memory storage, identifying and safely removing redundant or irrelevant memories, similar to how our brains are believed to prune unnecessary connections during sleep.

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

