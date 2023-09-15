using HdbscanSharp.Distance;
using HdbscanSharp.Runner;
using OwlCore.Extensions;
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
/// The Exocortex manages a collection of immutable memories, each represented by embedding vectors and creation timestamps.
/// This class is designed to be adaptable across various applications, allowing for interaction with different types of memory content (e.g., text, audio, visual).
/// </remarks>
/// <typeparam name="T">The type of raw content the memories hold.</typeparam>
public abstract partial class Exocortex<T>
{
    /// <summary>
    /// All memories created by the agent, in the order they were created.
    /// </summary>
    public HashSet<CortexMemory<T>> Memories { get; } = new HashSet<CortexMemory<T>>();

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
            // Maximum possible value for the threshold.
            double T_max = 1;

            // Minimum possible value for the threshold, taken from the LongTermDecayThreshold.
            double T_min = LongTermDecayThreshold;

            // Duration of long-term memory in hours.
            double D_lt = LongTermMemoryDuration.TotalHours;

            // Compute the logarithmic term. The "+ 1" inside the logarithm ensures that the value 
            // inside the log is always greater than 1, preventing negative results.
            double logTerm = Math.Log(1 + D_lt);

            // Calculate the ShortTermDecayThreshold using the formula. The result is designed to 
            // be between T_min and T_max based on the duration of the long-term memory.
            // The longer the duration, the closer the threshold will be to T_max.
            return T_min + (T_max - T_min) / (1 + logTerm);
        }
    }

    /// <summary>
    /// Gets or sets the threshold for determining when a long-term memory has effectively decayed.
    /// </summary>
    public double LongTermDecayThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets the duration for which a memory is considered as short-term before decaying to a specific threshold <see cref="ShortTermDecayThreshold"/>.
    /// </summary>
    public TimeSpan ShortTermMemoryDuration { get; set; } = TimeSpan.FromMinutes(25);

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
    public double RecalledWithContextMemoryWeight { get; set; } = 1.3;

    /// <summary>
    /// The weight used for reaction memories to a new core memory, with the recollections memories as added context.
    /// </summary>
    public double ReactionMemoryWeight { get; set; } = 1;

    /// <summary>
    /// Represents the number of dimensions used for UMAP (Uniform Manifold Approximation and Reduction), as well as HDBSCAN. Defaults to 3 dimensions, results in an average of about 3 clusters.
    /// </summary>
    /// <remarks>
    /// Adjusting the number of dimensions can have various implications:
    /// 
    /// - **2 Dimensions**: Often used for visualization purposes. Data is reduced to a 2D plane, which can lead to a potential loss of intricate data structures. This might result in some clusters merging or splitting.
    /// 
    /// - **Higher Dimensions (4, 5, ...)**: Increasing dimensions allows UMAP to preserve more of the local and global data structure. This can lead to a clearer separation of clusters in the higher-dimensional space. However, clustering algorithms might behave differently in higher dimensions due to the "curse of dimensionality". In higher dimensions, the distance between data points tends to become more uniform, making it harder to define dense regions.
    /// 
    /// The relationship between MaxRelatedRecollectionClusterMemories and NumberOfDimensions has shown to produce consistent clustering patterns. By multiplying MaxRelatedRecollectionClusterMemories by NumberOfDimensions, data is allowed to expand in the reduced space, potentially capturing more nuances. However, the exact behavior will depend on the inherent structure of your data and the interplay between UMAP and HDBSCAN. 
    /// 
    /// While maintaining this relationship can lead to consistent clustering, the exact number of clusters and their structure might vary based on the data and specific parameter values.
    /// 
    /// It's encouraged to experiment with different values and analyze the empirical results to get a clearer understanding of the effects.
    /// </remarks>
    public int NumberOfDimensions { get; set; } = 3;

    /// <summary>
    /// The maximum size of a cluster during recollection and consolidation. 
    /// </summary> 
    public int MaxRelatedRecollectionClusterMemories { get; set; } = 16;

    /// <summary>
    /// The max number of memories included in reaction formation.
    /// </summary>
    public int MaxRelatedReactionMemories { get; set; } = 20;

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

        // Inverse of recency.
        // A slight boost to the end of short-term memory will develop after a few years. Feature or bug? Adds attention to the end of the rolling context, may be good to keep.
        var nostalgia = 1 - recency;

        var typeWeight = memory.Type switch
        {
            CortexMemoryType.Core => CoreMemoryWeight,
            CortexMemoryType.Recollection => RecalledWithContextMemoryWeight,
            CortexMemoryType.Reaction => ReactionMemoryWeight,
            _ => throw new NotImplementedException(),
        };

        var finalWeight = ((nostalgia * relevance) + recency) * typeWeight;

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
        // Gather memories
        // Starting with the most recent short-term memories (including the new prompt), each short-term memory pulls in the most recent and relevant memories to it from the long-term.   
        var recollectionMemoriesWithWeights = new List<(CortexMemory<T>, double)>(ShortTermMemories.Select(x => (Memory: x, Score: ComputeMemoryWeight(newMemory, x.EmbeddingVector))));

        // TODO:
        // Recency isn't overtaking relevance at the very start of the memory stream, and we need more context than just what's related to the newest core memory.
        // This is the same trick we used when gathering memories for clustering, in order to bypass the limitation in the Umap library that only allows us to cluster based on relevance.
        // The way limits are set up will ensure that we always have only the highest weighted MaxRelatedRecollectionClusterMemories seen, and we always end up with a total of MaxRelatedRecollectionClusterMemories.
        // but some replaced with memories that are related to other short-term memories, those which bear a higher weighted sum (recency, relevance, nostalgia, etc).
        // This also limits clustering to only cluster MaxRelatedRecollectionClusterMemories. Further investigation on the effects of this are required.
        foreach (var item in recollectionMemoriesWithWeights.ToArray())
        {
            // Find long-term memories that are weighted higher.
            var relatedToShortTermMemory = LongTermMemories
                .Select(x => (Memory: x, Score: ComputeMemoryWeight(item.Item1, x.EmbeddingVector)))
                .OrderByDescending(x => x.Score)
                .Take(MaxRelatedRecollectionClusterMemories)
                .OrderBy(x => x.Memory.CreationTimestamp);

            foreach (var related in relatedToShortTermMemory)
                recollectionMemoriesWithWeights.Add(related);
        }

        // Apply limits and sorting
        // Grab a broad set of memories to reduce and cluster back into NumberOfDimensions via Umap / Hdbscan
        var recollectionMemories = recollectionMemoriesWithWeights
                .Select(x => (Memory: x.Item1, Score: x.Item2))
                .OrderByDescending(x => x.Score)
                .Take(MaxRelatedRecollectionClusterMemories * NumberOfDimensions)
                .Select(x => x.Memory)
                .OrderBy(x => x.CreationTimestamp)
                .Distinct();

        if (recollectionMemories.Any())
        {
            // Generate embeddings for all related memories
            float[][] embeddings = recollectionMemories.Select(x => x.EmbeddingVector.Select(v => (float)v).ToArray()).ToArray();

            // UMAP Reduction
            // TODO: UMAP and clustering should be based on the combined memory curve, not just relevancy.
            // This is a limitation of the Umap library being used.
            var umap = new Umap((x, y) => ComputeCosineSimilarity(x, y), dimensions: NumberOfDimensions, numberOfNeighbors: 1);
            var numberOfEpochs = umap.InitializeFit(embeddings);
            for (var i = 0; i < numberOfEpochs; i++)
                umap.Step();

            // Create reduced memories we can cluster.
            var recollectionMemoriesWithReducedDimensions = umap.GetEmbedding().Select((x, i) => new ReducedCortexMemory<T>(x.Select(x => (double)x).ToArray(), recollectionMemories.ElementAt(i))).ToArray();

            // Cluster memories
            // Memory clusters are similar to the prompt but different from each other.
            // The summaries reflect that, all rooted in the same prompt but augmented with a slightly different context.
            // Since recollections are short-term memories too, the AI sees all of them and reads between the lines, provided enough information.
            var clusterResult = HdbscanRunner.Run(new HdbscanParameters<CortexMemory<T>>
            {
                DataSet = recollectionMemoriesWithReducedDimensions, // double[][] for normal matrix or Dictionary<int, int>[] for sparse matrix
                MinPoints = 1,
                MinClusterSize = MaxRelatedRecollectionClusterMemories / NumberOfDimensions,
                CacheDistance = false, // using caching for distance throws unexpectedly
                MaxDegreeOfParallelism = 0, // to indicate all threads, you can specify 0.
                DistanceFunction = new CortexMemoryDistanceSpace<T>()
            });

            var clusteredMemories = recollectionMemoriesWithReducedDimensions.Zip(clusterResult.Labels, (memory, label) => (Memory: memory, Label: label)).ToList();

            foreach (var batchOfClusters in clusterResult.Labels.Distinct().Batch(3))
            {
                var results = await batchOfClusters.InParallel(async cluster =>
                {
                    // Skip noise points
                    if (cluster == -1)
                        return null;

                    // Retrieve original (non-reduced) memories in cluster
                    var clusterMemories = clusteredMemories
                        .Where(x => x.Label == cluster)
                        .Select(x => (Memory: x.Memory, Score: ComputeMemoryWeight(x.Memory, rawMemoryEmbedding)))
                        .OrderByDescending(x => x.Score)
                        .Take(MaxRelatedRecollectionClusterMemories)
                        .OrderBy(x => x.Memory.CreationTimestamp)
                        .Select(x => x.Memory is ReducedCortexMemory<T> reduced ? reduced.OriginalMemory : x.Memory)
                        .ToList();

                    if (clusterMemories.Count == 0)
                        return null;

                    var recollectionMemory = await SummarizeMemoryInNewContext(newMemory, clusterMemories);
                    var recollectionMemoryEmbedding = await GenerateEmbeddingAsync(recollectionMemory);
                    var memoryOfRecollection = new RecollectionCortexMemory<T>(recollectionMemory, recollectionMemoryEmbedding, clusterMemories);

                    Memories.Add(memoryOfRecollection);
                    return memoryOfRecollection;
                });

                foreach (var item in results)
                    if (item is not null)
                        yield return item;
            }
        }

        // ---------------
        // Reaction memory
        // ---------------
        // Create final reaction to the new memory, but with recent internal reflections.
        // Recency weights ensure recent recollections are prioritized over old ones.
        // Relevance weights ensure we can filter through large volumes of incoming information, as well as clusters with no useful information.
        
        // Do not begin iteration with memories already added, or it may struggle to recall memories older than the ones provided.
        // By iterating ShortTermMemories but comparing to all memories, it effectively replaces new memories with an older when the computed memory weight (recency, relevancy, nostalgia, etc) is higher than the original.
        // For the final reaction, we take `MaxRelatedReactionMemories` of the most recent memories, and starting with memories means they could be included regardless of their weights, if they're newer than the found memories.
        IEnumerable<CortexMemory<T>> reactionMemories = new HashSet<CortexMemory<T>>();

        // Gather memories
        // Clusters are formed from all long-term memories using similarity to short-term memories.
        // TODO:        
        // Recency isn't overtaking relevance at the very start of the memory stream, and we need more context than just what's related to the newest core memory.
        // This is the same trick we used when gathering memories for clustering, in order to bypass the limitation in the Umap library that only allows us to cluster based on relevance.
        // The way limits are set up will ensure that we always have MaxRelatedReactionMemories grabbed from short-term, and we always end up with MaxRelatedReactionMemories
        // but some replaced with memories that are related to other short-term memories, those which bear a higher weighted sum (recency, relevance, nostalgia, etc).
        foreach (var memory in ShortTermMemories)
        {
            var relatedToShortTermMemory = Memories
                .Select(x => (Memory: x, Score: ComputeMemoryWeight(x, memory.EmbeddingVector)))
                .OrderByDescending(x => x.Score)
                .Take(MaxRelatedReactionMemories)
                .Select(x => x.Memory);

            foreach (var related in relatedToShortTermMemory)
                ((HashSet<CortexMemory<T>>)reactionMemories).Add(related);
        }

        // Apply limits and sorting to memories in hashmap
        reactionMemories = reactionMemories
                .OrderByDescending(x => x.CreationTimestamp)
                .Take(MaxRelatedReactionMemories)
                .OrderBy(x => x.CreationTimestamp);

        var reaction = await ReactToMemoryAsync(newMemory, reactionMemories);
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
