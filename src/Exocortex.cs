using ClusterF_ck.DBSCAN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents the Exocortex, a generative remembrance agent that simulates the human brain's memory recall 
/// and consolidation processes. The Exocortex operates on a "rolling context" mechanism, ensuring that the most 
/// recent and relevant memories are prioritized, mimicking the human brain's ability to keep track of an 
/// ongoing conversation by constantly updating its understanding based on new information.
/// </summary>
/// <remarks>
/// The Exocortex manages a collection of memories, each represented by embedding vectors and creation timestamps.
/// 
/// Key Features:
/// 1. **Memory Decay**: Memories in the Exocortex decay over time, reflecting the human tendency to recall recent memories more vividly than older ones. This decay is modeled using an an intricate balance between duration, decay rate and decay thresholds for both short-term and long-term memory.
/// 3. **Memory Clustering**: Memories are grouped based on their content similarity. Within each cluster, a representative memory is chosen based on both its relevance to the current query and its recency.
/// 4. **Memory Retrieval**: The act of recalling a memory can modify its context and interpretation, reflecting the plastic nature of human memories. This process ensures that past memories are continually reframed in light of new experiences.
/// 
/// This class is designed to be adaptable across various applications, allowing for interaction with different types of memory content (e.g., text, audio, visual).
/// </remarks>
/// <typeparam name="T">The type of raw content the memories hold.</typeparam>
public abstract partial class Exocortex<T>
{
    /// <summary>
    /// All memories created by the agent, in the order they were created.
    /// </summary>
    public SortedSet<CortexMemory<T>> Memories { get; } = new SortedSet<CortexMemory<T>>();

    /// <summary>
    /// Short-term memory decay rate (Exponential decay)
    /// </summary>
    public double ShortTermDecayRate => -Math.Log(ShortTermDecayThreshold) / ShortTermMemoryDuration.TotalHours;

    /// <summary>
    /// Long-term memory decay rate (Reversed Logarithmic decay)
    /// </summary>
    public double LongTermDecayRate => (Math.Exp(1 - LongTermDecayThreshold) - 1) / LongTermMemoryDuration.TotalHours;

    /// <summary>
    /// Gets the threshold for determining when a short-term memory has effectively decayed.
    /// The short-term decay threshold is a function of the duration of long-term memories. As the 
    /// oldest memories in the system age, this threshold decreases, reflecting the idea that 
    /// the boundary between short-term and long-term recall becomes more forgiving.
    /// </summary>
    public double ShortTermDecayThreshold
    {
        get
        {
            // Function: Logarithmic function of long-term memory duration
            // Rationale: Represents the point at which short-term memories start 
            // transitioning to long-term memories. As more long-term memories accumulate, 
            // the threshold decreases, leading short-term memories to decay faster and long-term memories to decay slower.
            const double T_max = 1;
            double T_min = LongTermDecayThreshold;
            double D_lt = LongTermMemoryDuration.TotalHours;

            return T_max - (T_max - T_min) * Math.Log(1 + D_lt);
        }
    }

    /// <summary>
    /// Gets or sets the threshold for determining when a long-term memory has effectively decayed.
    /// </summary>
    public double LongTermDecayThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets the duration for which a memory is considered as short-term before decaying to a specific threshold <see cref="ShortTermDecayThreshold"/>.
    /// </summary>
    public TimeSpan ShortTermMemoryDuration { get; set; } = TimeSpan.FromMinutes(30);

    /// <summary>
    /// Gets or sets the duration for which a memory remains in long-term storage before decaying to a specific threshold <see cref="LongTermDecayThreshold"/>.
    /// </summary>
    public TimeSpan LongTermMemoryDuration
    {
        get
        {
            if (!Memories.Any())
            {
                return TimeSpan.Zero; // Return zero duration if there are no memories
            }

            // If we end up with a perf bottleneck we'll create a custom collection for Memories and adjust this value when the oldest memory changes.
            var oldestMemoryTimestamp = Memories.Min(memory => memory.CreationTimestamp);
            return DateTime.Now - oldestMemoryTimestamp;
        }
    }

    /// <summary>
    /// The weight used for the raw memory content provided to <see cref="AddMemoryAsync(T)"/>.
    /// </summary>
    public double CoreMemoryWeight { get; set; } = 1;

    /// <summary>
    /// The weight used for summary memories of the conversation where old and new context are combined.
    /// </summary>
    /// <remarks>
    /// The system should emphasize memory summaries (recollections) over core memories to provide a more concise and streamlined context.
    /// While core memories contain dense information, the recollections offer a summarized view, making them more suitable for quick 
    /// recall and relevance in ongoing conversations.
    /// </remarks>
    public double RecalledWithContextMemoryWeight { get; set; } = 2.5;

    /// <summary>
    /// The weight used for reaction memories to a new core memory, with the recollections memories as added context.
    /// </summary>
    public double ReactionMemoryWeight { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of related memories to recall when recalling and summarizing the context of a new memory.
    /// </summary>
    public int ThoughtDepth { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of related memories to recall in light of a new memory.
    /// </summary>
    public int ThoughtBreadth { get; set; } = 25;

    /// <summary>
    /// A value between 0 and 1 that indicates how similar the memories in a cluster are.
    /// </summary>
    public double MemoryClusterSimilarity { get; set; } = 0.75;

    /// <summary>
    /// Defines how the Exocortex should rewrite memories under the context of related memories.
    /// </summary>
    /// <param name="memory">The raw memory being experienced.</param>
    /// <param name="relatedMemories">The memories related to this new experience.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    public abstract Task<T> SummarizeMemoryInNewContext(CortexMemory<T> memory, IEnumerable<CortexMemory<T>> relatedMemories, CancellationToken cancellationToken = default);

    /// <summary>
    /// Defines how the Exocortex reacts to the train of thought spawned by a memory.
    /// </summary>
    /// <param name="memory">The raw memory being experienced.</param>
    /// <param name="relatedMemories">The memories related to this new experience.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    public abstract Task<T> ReactToMemoryAsync(CortexMemory<T> memory, IEnumerable<CortexMemory<T>> relatedMemories, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates an embedding vector for a given memory content.
    /// Used for computing similarities between memories.
    /// </summary>
    /// <param name="memoryContent">The content to generate an embedding for.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    /// <returns>A vector representing the content.</returns>
    public abstract Task<double[]> GenerateEmbeddingAsync(T memoryContent, CancellationToken cancellationToken = default);

    /// <summary>
    /// Computes the cosine similarity between two vectors. Normalized to [0, 1].
    /// This can be used to determine how similar two memories are.
    /// </summary>
    public static double ComputeCosineSimilarity(double[] vector1, double[] vector2)
    {
        double dotProduct = vector1.Zip(vector2, (a, b) => a * b).Sum();
        double magnitude1 = Math.Sqrt(vector1.Sum(a => a * a));
        double magnitude2 = Math.Sqrt(vector2.Sum(b => b * b));
        double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

        // Normalize to [0, 1]
        return (1 + cosineSimilarity) / 2;
    }

    /// <summary>
    /// Computes the recency score of a memory based on its creation timestamp using the decay model.
    /// </summary>
    /// /// <param name="creationTimestamp">The timestamp when the memory was created.</param>
    /// <returns>The recency score ranging from 0 (completely forgotten) to 1 (fully remembered).</returns>
    public double ComputeRecencyScore(DateTime creationTimestamp)
    {
        // Computes the recency score of a memory based on its creation timestamp 
        // using either the exponential decay model (short-term memory) or the 
        // reversed logarithmic decay model (long-term memory).

        double currentTime = (DateTime.Now - creationTimestamp).TotalHours;

        if (currentTime <= ShortTermMemoryDuration.TotalHours)
        {
            // Exponential decay for short-term memory
            return Math.Exp(-ShortTermDecayRate * currentTime);
        }
        else
        {
            // Reversed Logarithmic decay for long-term memory
            double decayRate = LongTermDecayRate;
            double x = LongTermMemoryDuration.TotalHours;
            return 1 - Math.Log(1 + decayRate * currentTime) / Math.Log(1 + decayRate * x);
        }
    }

    /// <summary>
    /// Retrieves and ranks memories relevant to a given query content based on their weighted scores.
    /// </summary>
    /// <param name="embedding">The embedding vector of the query/content for which related memories are to be recalled.</param>
    /// <returns>An ordered collection of memories, ranked by their weighted scores, which account for relevance, recency, and type.</returns>
    /// <remarks>
    /// This method identifies and prioritizes memories that are most relevant to a given query content. The prioritization is based on:
    /// 1. Relevance: Determined by the cosine similarity between each memory's embedding and the query embedding. Memories that are more similar to the query have higher relevance scores.
    /// 2. Recency: Modeled by the forgetting curve, which simulates the human tendency to recall recent memories more vividly than older ones. The recency of a memory diminishes over time.
    /// 3. Type: Different types of memories (e.g., core, recalled with context, reaction) can have different inherent weights, influencing their overall score.
    /// 
    /// The method begins by prioritizing very recent memories. It then clusters memories based on their similarities and selects a representative memory from each cluster. 
    /// The representative memory is chosen based on its proximity to the cluster's centroid and its weighted score.
    /// 
    /// Memories are grouped based on their content similarity. Within each cluster, a representative memory is chosen 
    /// based on both its relevance to the current query and its recency. This representative memory offers a 
    /// consolidated view of similar memories, enabling the Exocortex to augment the short-term context with relevant 
    /// long-term recollections without overwhelming the system with redundant information.
    /// </remarks>
    public IEnumerable<CortexMemory<T>> WeightedMemoryRecall(double[] embedding)
    {
        // This will hold the final set of representative memories
        var representativeMemories = new List<CortexMemory<T>>();

        // Prioritize very recent memories
        var recentMemories = Memories
            .OrderByDescending(m => ComputeMemoryWeight(m, embedding))
            .Take(10);

        representativeMemories.AddRange(recentMemories);

        // Get clusters of similar memories
        var clusters = ClusterSimilarMemories(Memories, MemoryClusterSimilarity);

        foreach (var cluster in clusters)
        {
            // For each memory in the cluster, compute a cluster-relative weight based on its relevance and recency
            var weightedMemories = cluster.Points
                .Select(memory => (Memory: memory, Score: ComputeMemoryWeight(memory, embedding)))
                .OrderBy(tuple => tuple.Score); // Order memories by their score

            // Pick the memory with the highest score as the representative for the cluster
            if (weightedMemories.Any())
                representativeMemories.Add(weightedMemories.First().Memory);
        }

        return representativeMemories.Distinct();
    }

    /// <summary>
    /// Computes the weighted score of a given memory based on its relevance to a query, its recency, and its type.
    /// </summary>
    /// <param name="memory">The memory whose weight is to be computed.</param>
    /// <param name="queryEmbedding">The embedding vector of the query/content being compared against the memory.</param>
    /// <returns>The computed weighted score of the memory.</returns>
    /// <remarks>
    /// The weight of a memory is determined by three factors:
    /// 1. Relevance: Measured by the cosine similarity between the memory's embedding and the query embedding.
    /// 2. Recency: Modeled by the forgetting curve, which captures how the strength of a memory decays over time.
    /// 3. Type: Different types of memories (e.g., core, recalled with context, reaction) might have different inherent weights.
    /// This method combines these factors to produce a composite weight for the memory.
    /// </remarks>
    public double ComputeMemoryWeight(CortexMemory<T> memory, double[] queryEmbedding)
    {
        var relevance = ComputeCosineSimilarity(queryEmbedding, memory.EmbeddingVector);
        var recency = ComputeRecencyScore(memory.CreationTimestamp);
        var typeWeight = memory.Type switch
        {
            CortexMemoryType.Core => CoreMemoryWeight,
            CortexMemoryType.RecalledWithContext => RecalledWithContextMemoryWeight,
            CortexMemoryType.Reaction => ReactionMemoryWeight,
            _ => throw new NotImplementedException(),
        };

        return relevance * recency * typeWeight;
    }


    /// <summary>
    /// Clusters similar memories together based on their embedding similarity.
    /// Within each cluster, memories are weighted by their recency.
    /// </summary>
    /// <param name="memories">The collection of memories to be clustered.</param>
    /// <param name="similarityThreshold">The threshold above which memories are considered similar and clustered together.</param>
    /// <returns>A sequence of clusters where each cluster is a sequence of similar memories.</returns>
    /// <remarks>
    /// This method uses a simple clustering approach where each memory is compared to existing clusters.
    /// If its similarity to a cluster (based on the weighted average embedding of the cluster) exceeds a 
    /// specified threshold, the memory is added to that cluster. Otherwise, a new cluster is created.
    /// <para/>
    /// 
    /// The use of embeddings allows for a more nuanced comparison between memories, capturing the 
    /// semantic essence of each memory.
    /// <para/>
    /// 
    /// By clustering memories, the Exocortex can identify patterns and groupings in the stored data, 
    /// which can be useful for various memory retrieval and analysis tasks.
    /// </remarks>
    public List<DBSCluster<CortexMemory<T>, CortexMemoryDistanceSpace<T>>> ClusterSimilarMemories(IEnumerable<CortexMemory<T>> memories, double similarityThreshold)
    {
        var clusterConfig = new DBSConfig(eps: similarityThreshold, minPts: 1, returnNoise: true);

        return DBSCAN.Cluster<CortexMemory<T>, CortexMemoryDistanceSpace<T>>(memories.ToArray().AsSpan(), clusterConfig, new CortexMemoryDistanceSpace<T>(this));
    }

    /// <summary>
    /// Computes a weighted average embedding for a given set of memories.
    /// Each memory's embedding is weighted by its recency, which means
    /// more recent memories have a stronger influence on the average.
    /// </summary>
    /// <param name="cluster">The set of memories for which the weighted average embedding should be computed.</param>
    /// <returns>The weighted average embedding. If the cluster is empty, returns null.</returns>
    /// <remarks>
    /// Embeddings are vector representations of data that capture its semantic content. By averaging 
    /// embeddings (and weighting by recency), we can get a representation that captures the "center" 
    /// or "essence" of a group of memories.
    /// 
    /// This method is particularly useful in the clustering process, allowing us to determine the 
    /// similarity of a memory to an existing cluster.
    /// </remarks>
    public double[]? ComputeWeightedAverageEmbedding(IEnumerable<CortexMemory<T>> cluster)
    {
        // If cluster is empty, return null.
        if (!cluster.Any())
            return null;

        int embeddingLength = cluster.First().EmbeddingVector.Length;
        double[] averagedEmbedding = new double[embeddingLength];
        double totalWeight = 0;

        foreach (var memory in cluster)
        {
            double weight = ComputeRecencyScore(memory.CreationTimestamp);
            totalWeight += weight;

            for (int i = 0; i < embeddingLength; i++)
            {
                averagedEmbedding[i] += memory.EmbeddingVector[i] * weight;
            }
        }

        // Normalize by total weight.
        for (int i = 0; i < embeddingLength; i++)
        {
            averagedEmbedding[i] /= totalWeight;
        }

        return averagedEmbedding;
    }

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    public Task AddMemoryAsync(T newMemoryContent) => AddMemoryAsync(newMemoryContent, DateTime.Now);

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    /// <param name="creationTimestamp">The <see cref="DateTime"/> this memory occurred at.</param>
    public async Task AddMemoryAsync(T newMemoryContent, DateTime creationTimestamp)
    {
        // Recall memories related to this new content
        var rawMemoryEmbedding = await GenerateEmbeddingAsync(newMemoryContent);
        var recollections = WeightedMemoryRecall(rawMemoryEmbedding).Take(ThoughtBreadth);

        var newMemory = new CortexMemory<T>(newMemoryContent, rawMemoryEmbedding)
        {
            CreationTimestamp = creationTimestamp,
            Type = CortexMemoryType.Core,
        };

        Memories.Add(newMemory);

        // Remember the act of recalling these memories, and roll reflections from one recollection to the next.
        var allRelevantMemories = new Dictionary<DateTime, CortexMemory<T>>(recollections.ToDictionary(x => x.CreationTimestamp));

        foreach (var memory in recollections)
        {
            // Recall and deduplicate memories related to our recollections.
            var relatedRecollections = WeightedMemoryRecall(memory.EmbeddingVector)
                .Except(new[] { memory })
                .OrderBy(x => x.CreationTimestamp)
                .Take(ThoughtDepth);

            foreach (var item in relatedRecollections)
            {
                if (allRelevantMemories.ContainsKey(item.CreationTimestamp))
                    continue;

                allRelevantMemories[item.CreationTimestamp] = item;
            }
        }

        // Interpret past memory + recollections about the new memory
        var relevantMemories = allRelevantMemories.Values
            .Take(ThoughtDepth)
            .OrderBy(x => x.CreationTimestamp)
            .ToArray();

        if (relevantMemories.Any())
        {
            // Roughly emulates the act of remembering and reflecting on thoughts before responding.
            // Context is rolled from the original memory, through a timeline of the most relevant and recent memories, and out into a "final thought".
            var recollectionMemory = await SummarizeMemoryInNewContext(newMemory, relevantMemories);
            var recollectionMemoryEmbedding = await GenerateEmbeddingAsync(recollectionMemory);

            var memoryOfRecollection = new CortexMemory<T>(recollectionMemory, recollectionMemoryEmbedding)
            {
                CreationTimestamp = DateTime.Now,
                Type = CortexMemoryType.RecalledWithContext,
            };

            allRelevantMemories.Add(memoryOfRecollection.CreationTimestamp, memoryOfRecollection);
            Memories.Add(memoryOfRecollection);
        }

        // Create final reaction to the new memory, but with recent internal reflections.
        // Recency weights ensure recent recollections are prioritized over old ones.
        // Relevance weights ensure we can filter through large volumes of incoming information.
        var reaction = await ReactToMemoryAsync(newMemory, allRelevantMemories.Values.Take(ThoughtDepth).OrderBy(x => x.CreationTimestamp));
        var reactionEmbedding = await GenerateEmbeddingAsync(reaction);

        var reactionMemory = new CortexMemory<T>(reaction, reactionEmbedding)
        {
            CreationTimestamp = DateTime.Now,
            Type = CortexMemoryType.Reaction,
        };

        Memories.Add(reactionMemory);
    }
}
