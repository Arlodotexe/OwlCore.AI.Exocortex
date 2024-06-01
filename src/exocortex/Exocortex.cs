using HdbscanSharp.Distance;
using HdbscanSharp.Hdbscanstar;
using HdbscanSharp.Runner;
using OwlCore.Extensions;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OwlCore.Storage.System.IO;
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
    /// The memories currently available to use by the agent, with user preferences.
    /// </summary>
    public IEnumerable<CortexMemory<T>> ActiveMemories => Memories.Where(x => x.CreationTimestamp <= PresentDateTime);

    /// <summary>
    /// All memories available to the agent.
    /// </summary>
    public HashSet<CortexMemory<T>> Memories { get; set; } = new HashSet<CortexMemory<T>>();

    /// <summary>
    /// All short term <see cref="ActiveMemories"/> within the <see cref="ShortTermMemoryDuration"/>.
    /// </summary>
    public IEnumerable<CortexMemory<T>> ShortTermMemories => ActiveMemories.Where(x => PresentDateTime - x.CreationTimestamp <= ShortTermMemoryDuration);

    /// <summary>
    /// All short term <see cref="ActiveMemories"/> within the <see cref="LongTermMemoryDuration"/>.
    /// </summary>
    public IEnumerable<CortexMemory<T>> LongTermMemories => ActiveMemories.Where(x => PresentDateTime - x.CreationTimestamp >= ShortTermMemoryDuration);

    /// <summary>
    /// The current date and time used for new memories.
    /// </summary>
    public DateTime PresentDateTime => CustomPresentDateTime ?? DateTime.Now;

    /// <summary>
    /// If set, the Exocortex will behave as though it is the current time. Newer memories will not be reachable.
    /// </summary>
    public DateTime? CustomPresentDateTime { get; set; }

    /// <summary>
    /// Gets the threshold for determining when a short-term memory has effectively decayed.
    /// The short-term decay threshold is dynamically computed based on the duration of the system's long-term memories.
    /// As the oldest memories in the system age, this threshold decreases, reflecting the idea that 
    /// the boundary between short-term and long-term recall becomes more forgiving. The threshold ranges between 
    /// T_min and T_max, which are defined in relation to the LongTermDecayThreshold.
    /// </summary>
    public float ShortTermDecayThreshold
    {
        get
        {
            // Maximum possible value for the short-term decay threshold. 
            float T_max = 1 - LongTermDecayThreshold;

            // Minimum possible value for the short-term decay threshold, 
            float T_min = LongTermDecayThreshold;

            // Duration of the oldest long-term memory in hours.
            float D_lt = (float)LongTermMemoryDuration.TotalHours;

            // Calculate the ShortTermDecayThreshold using an exponential decay formula.
            // The result is designed to be between T_min and T_max based on the duration 
            // of the oldest long-term memory. The longer this duration, the closer the 
            // threshold will be to T_min.
            return T_min + T_max * (float)Math.Exp(-0.00001 * D_lt);
        }
    }

    /// <summary>
    /// Gets or sets the threshold for determining when a long-term memory has effectively decayed.
    /// </summary>
    public float LongTermDecayThreshold { get; set; } = 0.01f;

    /// <summary>
    /// Gets the duration for which a memory is considered as short-term before decaying to a specific threshold <see cref="ShortTermDecayThreshold"/>.
    /// </summary>
    public TimeSpan ShortTermMemoryDuration { get; set; } = TimeSpan.FromHours(8);

    /// <summary>
    /// Gets or sets the duration for which a memory remains in long-term storage before decaying to a specific threshold <see cref="LongTermDecayThreshold"/>.
    /// </summary>
    public TimeSpan LongTermMemoryDuration
    {
        get
        {
            if (!ActiveMemories.Any())
            {
                return TimeSpan.Zero; // Return zero duration if there are no memories
            }

            // If we end up with a perf bottleneck we'll create a custom collection for Memories and adjust this value when the oldest memory changes.
            var oldestMemoryTimestamp = ActiveMemories.Min(memory => memory.CreationTimestamp);
            return PresentDateTime - oldestMemoryTimestamp;
        }
    }

    /// <summary>
    /// A weight between 0 and 1 used for the raw memory content provided to <see cref="AddMemoryAsync(T)"/>.
    /// </summary>
    public double CoreMemoryWeight { get; set; } = 1;

    /// <summary>
    /// A weight between 0 and 1 used for summary memories of the conversation where old and new context are combined.
    /// </summary>
    public double RecalledWithContextMemoryWeight { get; set; } = 1;

    /// <summary>
    /// A weight between 0 and 1 used for reaction memories to a new core memory, with the recollections memories as added context.
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
    /// It's encouraged to experiment with different values and analyze the empirical results to get a clearer understanding of the effects.
    /// </remarks>
    public int NumberOfDimensions { get; set; } = 3;

    /// <summary>
    /// Require memories to rank above this threshold to be used for working memory.
    /// </summary>
    /// <remarks>
    /// The value should be near the current <see cref="ShortTermDecayThreshold"/>, either above or below depending on your needs.
    /// </remarks>
    public double WorkingRecollectionMemoryWeightThreshold => ShortTermDecayThreshold + ((1 - ShortTermDecayThreshold) * WorkingRecollectionMemoryDistanceThreshold);

    /// <summary>
    /// Represents the distance into the short-term memory that a long-term memory must be weighted in order to be included as a working memory. The value should be near the current <see cref="ShortTermDecayThreshold"/>, either above or below depending on your needs.
    /// </summary>
    /// <remarks>
    /// <para/> Memories with recency weights above the ShortTermDecayThreshold are considered "short-term" memories.
    /// <para/> When using these to pull long-term memories, the nostalgia curve allows high relevancy weight of old memories to overtake the low recency weight.
    /// <para/> The amount here determines distance into short-term memory a memory needs to be boosted (indirectly, how much relevancy should overtake recency) in order to be included in working memory.
    ///
    /// <para/>  In psychology and neuroscience, memory span is the longest list of items that a person can repeat back in correct order immediately after presentation on 50% of all trials.
    /// <para/> This is the "Magic Number 7, plug or minus 2", and is controlled here.
    /// <para/> In the exocortex, the number of memories retrieved over 100 runs should average out to 7-8 when set to 0.25. No matter what value is set, results are halfed roughly every 100 years.
    /// <para/> The probability of correct recall for information presented depends on the capabilities of the LLM being used, the number of memories recalled, and the size of each memory provided.
    /// </remarks>
    public double WorkingRecollectionMemoryDistanceThreshold { get; set; } = 0.25f;

    /// <summary>ww
    /// Require memories to rank above this threshold to be used for working memory.
    /// </summary>
    /// <remarks>
    /// The value should be near the current <see cref="ShortTermDecayThreshold"/>, either above or below depending on your needs.
    /// </remarks>
    public float WorkingReactionMemoryWeightThreshold => ShortTermDecayThreshold + ((1 - ShortTermDecayThreshold) * WorkingReactionMemoryDistanceThreshold);

    /// <summary>
    /// Represents the distance into the short-term memory that a long memory must be weighted in order to be included as a working memory.
    /// </summary>
    /// <remarks>
    /// <para/> Memories with recency weights above the ShortTermDecayThreshold are considered "short-term" memories.
    /// <para/> When using these to pull long-term memories, the nostalgia curve allows high relevancy weight of old memories to overtake the low recency weight.
    /// <para/> The amount here determines distance into short-term memory a memory needs to be boosted (indirectly, how much relevancy should overtake recency) in order to be included in working memory.
    ///
    /// <para/>  In psychology and neuroscience, memory span is the longest list of items that a person can repeat back in correct order immediately after presentation on 50% of all trials.
    /// <para/> This is the "Magic Number 7, plus or minus 2", and is controlled here.
    /// <para/> In the exocortex, the number of memories retrieved over 100 runs should average out to 7-8 when set to 0.25. No matter what value is set, results are halfed roughly every 100 years.
    /// <para/> The probability of correct recall for information presented depends on the capabilities of the LLM being used, the number of memories recalled, and the size of each memory provided.
    /// </remarks>
    public float WorkingReactionMemoryDistanceThreshold { get; set; } = 0.25f;

    /// <summary>
    /// Defines how the Exocortex should rewrite memories under the context of related memories.
    /// </summary>
    /// <param name="memory">The raw memory being experienced.</param>
    /// <param name="workingMemories">The memories that have been deemed relevant or recent enough to be used for summarization.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    public abstract Task<T> SummarizeMemoryInNewContext(CortexMemory<T> memory, IEnumerable<CortexMemory<T>> workingMemories, CancellationToken cancellationToken = default);

    /// <summary>
    /// Defines how the Exocortex reacts to the train of thought spawned by a memory.
    /// </summary>
    /// <param name="memory">The raw memory being experienced.</param>
    /// <param name="workingMemories">The memories that have been deemed relevant or recent enough to be used for summarization.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    public abstract Task<T> ReactToMemoryAsync(CortexMemory<T> memory, IEnumerable<CortexMemory<T>> workingMemories, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates an embedding vector for a given memory content.
    /// Used for computing similarities between memories.
    /// </summary>
    /// <param name="memoryContent">The content to generate an embedding for.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    /// <returns>A vector representing the content.</returns>
    public abstract Task<float[]> GenerateEmbeddingAsync(T memoryContent, CancellationToken cancellationToken = default);

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
        if (finalWeight > 1.0001 || finalWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

        return (float)finalWeight;
    }

    /// <summary>
    /// Computes the recency score of a memory based on its creation timestamp using the decay model.
    /// </summary>
    /// /// <param name="creationTimestamp">The timestamp when the memory was created.</param>
    /// <returns>The recency score ranging from 0 (completely forgotten) to 1 (fully remembered).</returns>
    public double ComputeRecencyWeight(DateTime creationTimestamp)
    {
        var maxLifetime = LongTermMemoryDuration.TotalHours;
        double baseE = Math.E;

        // Computes the recency score of a memory based on its creation timestamp 
        // using either the exponential decay model (short-term memory) or the 
        // reversed logarithmic decay model (long-term memory).
        double currentTime = (PresentDateTime - creationTimestamp).TotalHours;

        if (currentTime <= ShortTermMemoryDuration.TotalHours)
        {
            // Exponential decay for short-term memory
            var shortTermDecayRate = -Math.Log(ShortTermDecayThreshold) / ShortTermMemoryDuration.TotalHours;
            var finalWeight = Math.Exp(-shortTermDecayRate * currentTime);
            if (finalWeight > 1 || finalWeight < 0)
                throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

            return finalWeight;
        }
        else
        {
            // Reversed Logarithmic decay for long-term memory
            double a = LongTermDecayThreshold;
            double b = (maxLifetime - LongTermMemoryDuration.TotalHours) * (-Math.Log(baseE) / Math.Log(maxLifetime));

            var remainingLifetime = currentTime - maxLifetime;
            double finalWeight = a * Math.Pow(baseE, -b * remainingLifetime / maxLifetime);

            if (finalWeight > 1 || finalWeight < 0)
                throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

            return finalWeight;
        }
    }

    /// <summary>
    /// Computes the weighted score (between 0 and 1) of a given memory based on its relevance to a query, its recency, and its type.
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
    public virtual double ComputeFullMemoryWeight(CortexMemory<T> memory, float[] queryEmbedding)
    {
        var typeWeight = memory.Type switch
        {
            CortexMemoryType.Core => CoreMemoryWeight,
            CortexMemoryType.Recollection => RecalledWithContextMemoryWeight,
            CortexMemoryType.Reaction => ReactionMemoryWeight,
            _ => throw new NotImplementedException(),
        };

        var relevance = ComputeCosineSimilarity(memory.EmbeddingVectors, queryEmbedding);
        var recency = (float)ComputeRecencyWeight(memory.CreationTimestamp);

        // Relevance-based nostalgia is used to counter the weight of older memories that are still relevant.
        // The intersection point of recency and nostalgia curves.
        // This is a constant since the nostalgia curve is the inverse of the recency curve after a certain point.
        const float intersectionPoint = 0.5f;

        // Inverse of recency, starting at the intersection point with the recency curve.
        // No nostalgia boost is used on memories where it would reduce relevance (before the intersection point).
        // A slight boost to the end of short-term memory will develop after about a decade. Feature or bug? Adds attention to the end of the rolling context, may be good to keep.
        float? nostalgia = recency <= intersectionPoint ? (recency - 1.00001f) + 1 : null;

        // Merge these values as if they operate on the same one-dimensional axis (using addition)
        // Creates a symmetrical 1-dimensional value that cleanly represents 2 dimensions (via logical OR operation / addition).
        var relativeSumNostalgia = (nostalgia ?? 0) + relevance;

        // Merge these values as if they're being crossed on two orthogonal one-dimensional axis (using multiplication).
        // Creates a symmetrical 1-dimensional value that cleanly represents 2 dimensions (via logical AND operation / multiplication).
        var relativeCrossNostalgia = (nostalgia ?? -1.00001f) * relevance;

        // This is the remainder of the differences between 2 one-dimensional values with external symmetry (effectively, two 2-dimensional values with internal symmetry)
        // we can combine these and reduce by 1 dimension using the Pythagorean theorem
        var relativeNostalgia = Math.Sqrt(relativeSumNostalgia + relativeCrossNostalgia);

        var finalWeight = relativeNostalgia * typeWeight;

        MemoryJson.Add($"{{ {nameof(nostalgia)}: {nostalgia?.ToString() ?? "null"}, {nameof(relevance)}: {relevance}, {nameof(recency)}: {recency}, unix_timestamp: {((DateTimeOffset)memory.CreationTimestamp).ToUnixTimeMilliseconds()}, {nameof(finalWeight)}: {finalWeight} }},");

        // Using 4 as the limiter since 4 dimensions will naturally collapse back to 2 when reduced, and our operations are purely 2-dimensional.
        // We should never be able to exceed a value of 4.
        if (finalWeight > 4 || finalWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(finalWeight), "Memory weight out of range.");

        return finalWeight;
    }

    private List<string> MemoryJson = new List<string>();

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    public async IAsyncEnumerable<CortexMemory<T>> AddMemoryAsync(T newMemoryContent)
    {
        // ---------------
        // Core memory
        // ---------------
        // Recall memories related to this new content
        var rawMemoryEmbedding = await GenerateEmbeddingAsync(newMemoryContent);
        var newMemory = new CortexMemory<T>(newMemoryContent, rawMemoryEmbedding, PresentDateTime)
        {
            Type = CortexMemoryType.Core,
        };

        Memories.Add(newMemory);
        yield return newMemory;

        // ---------------
        // Recollection memory
        // ---------------
        // Gather memories
        // Starting with the most recent short-term memories (including the new prompt), each short-term memory pulls in the most recent and relevant memories to it from the long-term.
        var recollectionMemories = ShortTermMemories
            .SelectMany(stMemory => ActiveMemories.Select(ltMemory => new WorkingCortexMemory<T>(ltMemory, ComputeFullMemoryWeight(ltMemory, stMemory.EmbeddingVectors))))
                .GroupBy(x => x.WeighedMemory) // DistinctBy
                .Select(g => g.First())
            //.Where(x => x.Score >= WorkingRecollectionMemoryWeightThreshold)
            .ToList();

        var jsonInner = MemoryJson.Aggregate((x, y) => $"{x}{Environment.NewLine}{y}").TrimEnd(',');
        var json = $"[ {Environment.NewLine} {jsonInner} {Environment.NewLine} ]";
        var jsonRawBytes = Encoding.UTF8.GetBytes(json);

        var file = new SystemFile("D:\\source\\dotnet\\core\\LlmPlayground\\OwlCore.AI.Exocortex\\docs\\plotting\\file.json");

        var stream = await file.OpenStreamAsync(FileAccess.Write);
        stream.Seek(0, SeekOrigin.Begin);
        await stream.WriteAsync(jsonRawBytes, 0, jsonRawBytes.Length);
        stream.SetLength(jsonRawBytes.Length);

        stream.Dispose();

        if (recollectionMemories.Count > NumberOfDimensions) // Number of dimensions roughly determines number of clusters and their sizes. We need enough memories to do clustering.
        {
            // Generate embeddings for all related memories
            var dataPoints = recollectionMemories.Select(x => new CortexMemoryUmapDataPoint<T>(x)).ToArray();

            // UMAP Reduction
            var umap = new Umap<CortexMemoryUmapDataPoint<T>>((x, y) => (float)new CortexMemoryDistanceSpace<T>(this).ComputeDistance(0, 0, x, y), dimensions: NumberOfDimensions, numberOfNeighbors: NumberOfDimensions);
            var numberOfEpochs = umap.InitializeFit(dataPoints);
            for (var i = 0; i < numberOfEpochs; i++)
                umap.Step();

            // Create reduced memories we can cluster.
            var recollectionMemoriesWithReducedDimensions = umap.GetEmbedding().Select((x, i) => new ReducedCortexMemory<T>(x, recollectionMemories[i], PresentDateTime)).ToArray();

            // Cluster memories
            // Memory clusters are similar to the prompt but different from each other.
            // The summaries reflect that, all rooted in the same prompt but augmented with a slightly different context.
            // Since recollections are short-term memories too, the AI sees all of them and reads between the lines, provided enough information.
            var clusterResult = HdbscanRunner.Run(new HdbscanParameters<CortexMemory<T>>
            {
                DataSet = recollectionMemoriesWithReducedDimensions, // double[][] for normal matrix or Dictionary<int, int>[] for sparse matrix

                // MinPoints determines the minimum size of the "core" group of points which are closely packed together in the space. In the context of memory recall:
                // - A larger MinPoints value will result in fewer points being considered as core points, leading to fewer and larger clusters.
                //   This may simulate a more generalized memory recall where fewer but broader topics are recalled.
                // - A smaller MinPoints value will allow more points to be considered as core points, leading to more and smaller clusters.
                //   This might be useful for detailed and specific memory recall, as it will form more clusters representing nuanced memories.
                MinPoints = 3,

                // MinClusterSize defines the smallest size a group of points can be to be considered a cluster, rather than noise.
                // - A larger MinClusterSize will result in larger, but fewer clusters.
                //   This means your exocortex would prioritize forming significant memory clusters and disregarding less substantial ones.
                // - A smaller MinClusterSize allows the formation of smaller clusters, which could represent more specific memory recalls.
                //   However, too small a size could also pick up noise as valid clusters, which might lead to recalling irrelevant or less important memories.
                MinClusterSize = 3,
                CacheDistance = false, // using caching for distance throws unexpectedly
                MaxDegreeOfParallelism = 0, // to indicate all threads, you can specify 0.
                DistanceFunction = new CortexMemoryDistanceSpace<T>(this)
            });

            var clusteredMemories = recollectionMemoriesWithReducedDimensions.Zip(clusterResult.Labels, (memory, label) => (Memory: memory, Label: label)).ToList();

            foreach (var batchOfClusters in clusterResult.Labels.Distinct().Batch(1))
            {
                var results = await batchOfClusters.InParallel(async cluster =>
                {
                    // Skip noise points
                    if (cluster == -1)
                        return null;

                    // Retrieve original (non-reduced) memories in cluster
                    var clusterMemories = clusteredMemories
                        .Where(x => x.Label == cluster)
                        .Select(x => x.Memory is ReducedCortexMemory<T> reduced ? reduced.OriginalMemory : x.Memory)
                        .Cast<WorkingCortexMemory<T>>()
                        .Where(x => x.Score >= WorkingRecollectionMemoryWeightThreshold)
                        .ToList();

                    if (clusterMemories.Count < 2)
                        return null;

                    var recollectionMemory = await SummarizeMemoryInNewContext(newMemory, clusterMemories.OrderBy(x => x.CreationTimestamp));
                    var recollectionMemoryEmbedding = await GenerateEmbeddingAsync(recollectionMemory);
                    var memoryOfRecollection = new RecollectionCortexMemory<T>(recollectionMemory, recollectionMemoryEmbedding, clusterMemories, PresentDateTime);

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

        // By iterating ShortTermMemories but comparing to all memories, it effectively replaces new memories with an older one when the computed memory weight (recency, relevancy, nostalgia, etc) is higher than the original.
        // For the final reaction, we take `MaxRelatedReactionMemories` of the most recent memories, and starting with memories means they could be included regardless of their weights, if they're newer than the found memories.
        var reactionMemories = ShortTermMemories
            .SelectMany(stMem => ActiveMemories.Select(ltMem => new WorkingCortexMemory<T>(stMem, ComputeFullMemoryWeight(ltMem, stMem.EmbeddingVectors))))
            .Where(x => x.Score >= WorkingReactionMemoryWeightThreshold)
            .OrderBy(x => x.CreationTimestamp)
                .GroupBy(x => x.WeighedMemory) // DistinctBy
                .Select(g => g.First());

        ////////////////
        // NOTES
        /////////////
        // this is grabbing too many memories. It grabs a normal amount (magic number 7, +-2) for a single memory,
        // but iterating multiple short-term memories doesn't always grab the same memory.
        // The solution to this isn't straightfoward. We can either (or both):
        // - Find a way to grab the same memory for similar memories. Maybe check the similarity of memories we already have, and use 
        // - Reconsider how clustering fits into the picture. We need clustering in order to create that rolling context
        //   But perhaps it would be better suited as a way to consolidate down all the possible memories into just a few?
        //   It would still have the same effect, and could still be labeled as "recollection".
        var reaction = await ReactToMemoryAsync(newMemory, reactionMemories.OrderBy(x => x.CreationTimestamp));

        var reactionEmbedding = await GenerateEmbeddingAsync(reaction);

        var reactionMemory = new CortexMemory<T>(reaction, reactionEmbedding, PresentDateTime)
        {
            Type = CortexMemoryType.Reaction,
        };

        Memories.Add(reactionMemory);
        yield return reactionMemory;
    }
}
