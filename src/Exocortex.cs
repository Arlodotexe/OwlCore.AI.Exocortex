using HdbscanSharp.Distance;
using HdbscanSharp.Runner;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UMAP;

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
    /// All short term <see cref="Memories"/> within the <see cref="ShortTermMemoryDuration"/>.
    /// </summary>
    public IEnumerable<CortexMemory<T>> ShortTermMemories => Memories.Where(x => DateTime.Now - x.CreationTimestamp <= ShortTermMemoryDuration);

    /// <summary>
    /// All short term <see cref="Memories"/> within the <see cref="LongTermMemoryDuration"/>.
    /// </summary>
    public IEnumerable<CortexMemory<T>> LongTermMemories => Memories.Where(x => DateTime.Now - x.CreationTimestamp >= ShortTermMemoryDuration);

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
    public TimeSpan ShortTermMemoryDuration { get; set; } = TimeSpan.FromMinutes(3);

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
    public double RecalledWithContextMemoryWeight { get; set; } = 0.5;

    /// <summary>
    /// The weight used for reaction memories to a new core memory, with the recollections memories as added context.
    /// </summary>
    public double ReactionMemoryWeight { get; set; } = 1;

    /// <summary>
    /// The maximum number of memories recalled from long-term memory.
    /// </summary>
    public int LongTermMemoryRecallLimit { get; set; } = 25;

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
    public static float ComputeCosineSimilarity(float[] vector1, float[] vector2)
    {
        var dotProduct = vector1.Zip(vector2, (a, b) => a * b).Sum();
        var magnitude1 = Math.Sqrt(vector1.Sum(a => a * a));
        var magnitude2 = Math.Sqrt(vector2.Sum(b => b * b));
        var cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

        // Normalize to [0, 1]
        var finalWeight = (1 + cosineSimilarity) / 2f;
        if (finalWeight > 1 || finalWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

        return (float)finalWeight;
    }

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
        var finalWeight = (1 + cosineSimilarity) / 2;
        if (finalWeight > 1 || finalWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

        return finalWeight;
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
            var finalWeight = Math.Exp(-ShortTermDecayRate * currentTime);
            if (finalWeight > 1 || finalWeight < 0)
                throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

            return finalWeight;
        }
        else
        {
            // Reversed Logarithmic decay for long-term memory
            double decayRate = LongTermDecayRate;
            double x = LongTermMemoryDuration.TotalHours;

            var finalWeight = 1 - Math.Log(1 + decayRate * currentTime) / Math.Log(1 + decayRate * x);
            if (finalWeight > 1 || finalWeight < 0)
                throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

            return finalWeight;
        }
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
        var relevance = ComputeCosineSimilarity(memory.EmbeddingVector, queryEmbedding);
        var recency = ComputeRecencyScore(memory.CreationTimestamp);
        var typeWeight = memory.Type switch
        {
            CortexMemoryType.Core => CoreMemoryWeight,
            CortexMemoryType.Recollection => RecalledWithContextMemoryWeight,
            CortexMemoryType.Reaction => ReactionMemoryWeight,
            _ => throw new NotImplementedException(),
        };

        var finalWeight = relevance * recency * typeWeight;

        if (finalWeight > 2 || finalWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

        return finalWeight;
    }

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    public IAsyncEnumerable<CortexMemory<T>> AddMemoryAsync(T newMemoryContent) => AddMemoryAsync(newMemoryContent, DateTime.Now);

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    /// <param name="creationTimestamp">The <see cref="DateTime"/> this memory occurred at.</param>
    public async IAsyncEnumerable<CortexMemory<T>> AddMemoryAsync(T newMemoryContent, DateTime creationTimestamp)
    {
        // ---------------
        // Core memory
        // ---------------
        // Recall memories related to this new content
        var rawMemoryEmbedding = await GenerateEmbeddingAsync(newMemoryContent);
        var newMemory = new CortexMemory<T>(newMemoryContent, rawMemoryEmbedding)
        {
            CreationTimestamp = creationTimestamp,
            Type = CortexMemoryType.Core,
        };

        Memories.Add(newMemory);
        yield return newMemory;

        // ---------------
        // Recollection memory
        // ---------------
        // Remember the act of recalling these memories, and roll reflections from one recollection to the next.
        // Roughly emulates the act of remembering and reflecting on thoughts before responding.
        // Context is rolled from the original memory, through a timeline of the most relevant and recent memories, and out into a "final thought".
        var relatedMemories = LongTermMemories
                .Select(memory => (Memory: memory, Score: ComputeMemoryWeight(memory, rawMemoryEmbedding)))
                .OrderBy(tuple => tuple.Score)
                .Select(x => x.Memory); // Order memories by their score

        if (relatedMemories.Any())
        {
            // Generate embeddings for all related memories
            float[][] embeddings = relatedMemories.Select(x => x.EmbeddingVector.Select(v => (float)v).ToArray()).ToArray();

            // UMAP Reduction
            var umap = new Umap((x, y) => ComputeCosineSimilarity(x, y), dimensions: 5, numberOfNeighbors: 1);
            var numberOfEpochs = umap.InitializeFit(embeddings);
            for (var i = 0; i < numberOfEpochs; i++)
                umap.Step();

            // Create reduced memories we can cluster.
            var relatedMemoriesWithReducedDimensions = umap.GetEmbedding().Select((x, i) => new ReducedCortexMemory<T>(x.Select(x => (double)x).ToArray(), relatedMemories.ElementAt(i))).ToArray();

            // Cluster memories
            var clusterResult = HdbscanRunner.Run(new HdbscanParameters<CortexMemory<T>>
            {
                DataSet = relatedMemoriesWithReducedDimensions, // double[][] for normal matrix or Dictionary<int, int>[] for sparse matrix
                MinPoints = 1,
                MinClusterSize = 2,
                CacheDistance = false, // use caching for distance
                MaxDegreeOfParallelism = 0, // to indicate all threads, you can specify 0.
                DistanceFunction = new CortexMemoryDistanceSpace<T>() // See HdbscanSharp/Distance folder for more distance function
            });

            var clusteredMemories = relatedMemoriesWithReducedDimensions.Zip(clusterResult.Labels, (memory, label) => (Memory: memory, Label: label)).ToList();

            var tasks = new List<Task>();
            var results = new List<CortexMemory<T>>();

            foreach (var cluster in clusterResult.Labels.Distinct())
            {
                tasks.Add(Task.Run(async () =>
                {
                    // Skip noise points
                    if (cluster == -1)
                        return;

                    // Retrieve original (non-reduced) memory
                    // Order by memory creation time.
                    var clusterMemories = clusteredMemories
                        .Where(x => x.Label == cluster)
                        .Select(x => (Memory: x.Memory, Score: ComputeMemoryWeight(x.Memory, rawMemoryEmbedding)))
                        .OrderBy(x => x.Score)
                        .Take(LongTermMemoryRecallLimit)
                        .OrderBy(x => x.Memory.CreationTimestamp)
                        .Select(x => x.Memory is ReducedCortexMemory<T> reduced ? reduced.OriginalMemory : x.Memory);

                    var recollectionMemory = await SummarizeMemoryInNewContext(newMemory, clusterMemories);
                    var recollectionMemoryEmbedding = await GenerateEmbeddingAsync(recollectionMemory);
                    var memoryOfRecollection = new RecollectionCortexMemory<T>(recollectionMemory, recollectionMemoryEmbedding, clusterMemories);

                    Memories.Add(memoryOfRecollection);
                    results.Add(memoryOfRecollection);
                }));
            }

            await Task.WhenAll(tasks);

            foreach (var item in results)
                yield return item;
        }

        // ---------------
        // Reaction memory
        // ---------------
        // Create final reaction to the new memory, but with recent internal reflections.
        // Recency weights ensure recent recollections are prioritized over old ones.
        // Relevance weights ensure we can filter through large volumes of incoming information.
        var reaction = await ReactToMemoryAsync(newMemory, ShortTermMemories);
        var reactionEmbedding = await GenerateEmbeddingAsync(reaction);

        var reactionMemory = new CortexMemory<T>(reaction, reactionEmbedding)
        {
            CreationTimestamp = DateTime.Now,
            Type = CortexMemoryType.Reaction,
        };

        Memories.Add(reactionMemory);

        yield return reactionMemory;
    }
}
