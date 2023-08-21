using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents the Exocortex, a generative rememberance agent that mimics the human brain's memory recall and consolidation processes.
/// </summary>
/// <remarks>
/// The Exocortex manages a collection of memories, each represented by embedding vectors, importance scores, and creation timestamps.
/// <para/>
/// It continually reframes and consolidates old memories in light of new experiences, updating the context and interpretation of past events without overwriting the original memories.
/// This process resembles how recalling a human memory can change the way it is remembered and how the act of recalling can become a new memory itself.
/// <para/>
/// Memories in the Exocortex are subject to decay over time (recency), modeled as an exponential decay function. This reflects the human tendency
/// to recall recent memories more vividly than distant ones.
/// <para/>
/// The Exocortex uses a sophisticated mechanism to assign importance to memories, distinguishing significant experiences from mundane events,
/// akin to how the human mind assigns emotional weight to different memories.
/// <para/>
/// Each new memory is compared (using cosine similarity between embedding vectors) to existing memories to establish relevance. This emulates how
/// human memory retrieval is often triggered by related events or thoughts.
/// <para/>
/// This class is designed to interact with different types of memory (e.g., text, audio, visual), making it adaptable for various applications.
/// </remarks>
/// <typeparam name="T">The type of raw content the memories hold.</typeparam>
public abstract partial class Exocortex<T>
{
    /// <summary>
    /// All memories created by the agent, in the order they were created.
    /// </summary>
    public SortedSet<CortexMemory<T>> Memories { get; } = new SortedSet<CortexMemory<T>>();

    /// <summary>
    /// Gets or sets the threshold for determining when a short-term memory has effectively decayed.
    /// </summary>
    public double ShortTermDecayThreshold { get; set; } = 0.1;  // Default to 10%

    /// <summary>
    /// Gets or sets the threshold for determining when a long-term memory has effectively decayed.
    /// </summary>
    public double LongTermDecayThreshold { get; set; } = 0.01;  // Default to 1%

    /// <summary>
    /// Gets or sets the duration for which a memory is considered in short-term storage before decaying to a specific threshold.  Default to 1 minute.
    /// </summary>
    public TimeSpan ShortTermMemoryDuration { get; set; } = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Gets or sets the duration for which a memory remains in long-term storage before decaying to a specific threshold. Default to 1 hour.
    /// </summary>
    public TimeSpan LongTermMemoryDuration { get; set; } = TimeSpan.FromHours(1);

    /// <summary>
    /// Gets the decay rate for memories within the recent memory window (short-term memory).
    /// Represents the rapid decay characteristic of human working memory.
    /// </summary>
    /// <remarks>
    /// A higher value indicates a faster decay. By default, set to represent a rapid decay for recent memories.
    /// </remarks>
    public double ShortTermDecayRate => ComputeDecayRate(ShortTermMemoryDuration.TotalHours, ShortTermDecayThreshold);

    /// <summary>
    /// Gets the decay rate for memories outside of the recent memory window (long-term memory).
    /// Represents the slower decay characteristic of human long-term memory.
    /// </summary>
    /// <remarks>
    /// A higher value indicates a faster decay. By default, set to represent a more persistent, slower decay for older memories.
    /// </remarks>
    public double LongTermDecayRate => ComputeDecayRate(LongTermMemoryDuration.TotalHours, LongTermDecayThreshold);

    /// <summary>
    /// Gets or sets the number of related memories to recall when recalling and summarizing the context of a new memory.
    /// </summary>
    public int ThoughtDepth { get; set; } = 25;

    /// <summary>
    /// Gets or sets the number of related memories to recall in light of a new memory.
    /// </summary>
    public int ThoughtBreadth { get; set; } = 25;

    /// <summary>
    /// Gets or sets a boolean value that indicates if <see cref="ThoughtBreadth"/> is adjust automatically based on the average relevance of the top 10 memories.
    /// </summary>
    public bool AutoAdjustThoughtBreadth { get; set; } = false;

    /// <summary>
    /// Gets or sets a boolean value that indicates if <see cref="ThoughtDepth"/> is adjust automatically based on the average relevance of the top 10 memories.
    /// </summary>
    public bool AutoAdjustThoughtDepth { get; set; } = false;

    /// <summary>
    /// The maximum depth of <see cref="ThoughtDepth"/>.
    /// </summary>
    public int ThoughtDepthMax { get; set; } = 10;

    /// <summary>
    /// The maximum Breadth of <see cref="ThoughtBreadth"/>.
    /// </summary>
    public int ThoughtBreadthMax { get; set; } = 50;

    /// <summary>
    /// A value between 0 and 1 that indicates how similar the memories in a cluster are. The most recent memory in a cluster is always used.
    /// </summary>
    public double MemoryClusterSimilarity { get; set; } = 0.8;

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
    /// Computes the decay rate for a given duration and threshold.
    /// </summary>
    /// <param name="x">The desired duration (in hours) for the memory to decay to the specified threshold.</param>
    /// <param name="threshold">The recency score threshold representing the "effectively decayed" state.</param>
    /// <returns>The decay rate for the logistic decay function.</returns>
    /// <remarks>
    /// The inflection point (x0) of the logistic decay function is set to the midpoint of the specified duration.
    /// This ensures that the decay is centered around the midpoint of the duration, allowing for a balanced decay curve.
    /// </remarks>
    private double ComputeDecayRate(double x, double threshold)
    {
        // Setting the inflection point to the midpoint of the specified duration
        double x0 = x / 2;
        return Math.Log((1 / threshold) - 1) / (x - x0);
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
        return (1 + cosineSimilarity) / 2;
    }

    /// <summary>
    /// Computes the recency score of a memory based on its creation timestamp.
    /// This score models how 'fresh' or recent a memory is.
    /// </summary>
    public double ComputeRecencyScore(DateTime creationTimestamp)
    {
        var timeSpanSinceCreation = DateTime.Now - creationTimestamp;
        var minutesSinceCreation = timeSpanSinceCreation.TotalMinutes;

        // Determine which decay rate to use based on the age of the memory
        double decayRate = (timeSpanSinceCreation <= ShortTermMemoryDuration) ? ShortTermDecayRate : LongTermDecayRate;

        // Determine the inflection point (midpoint) of the curve based on the chosen decay rate
        double x0 = (decayRate == ShortTermDecayRate) ? ShortTermMemoryDuration.TotalMinutes / 2 : LongTermMemoryDuration.TotalMinutes / 2;

        // Logistic decay function
        var recencyScore = 1 / (1 + Math.Exp(decayRate * (minutesSinceCreation - x0)));

        return recencyScore;
    }

    /// <summary>
    /// Retrieves and ranks memories relevant to a given query content.
    /// </summary>
    /// <param name="embedding">The embeddings for this content, if available.</param>
    /// <returns>An ordered set of memories, ranked by relevance, importance, and recency.</returns>
    public IEnumerable<CortexMemory<T>> WeightedMemoryRecall(double[] embedding)
    {
        // This will hold the final set of representative memories
        var representativeMemories = new List<CortexMemory<T>>();

        // Prioritize very recent memories
        var recentMemories = Memories
            .OrderByDescending(m => ComputeRecencyScore(m.CreationTimestamp))
            .Where(x => x.Type != CortexMemoryType.RecalledWithContext)
            .Take(10);

        representativeMemories.AddRange(recentMemories);

        // Get clusters of similar memories
        var clusters = ClusterSimilarMemories(Memories, MemoryClusterSimilarity);

        foreach (var cluster in clusters)
        {
            // For each memory in the cluster, compute a weight based on its relevance and recency
            var weightedMemories = cluster
                .Select(memory =>
                {
                    var relevance = ComputeCosineSimilarity(embedding, memory.EmbeddingVector);
                    var recency = ComputeRecencyScore(memory.CreationTimestamp);

                    // Combine relevance and recency to get a score
                    var score = relevance * recency;

                    return (Memory: memory, Score: score);
                })
                .OrderByDescending(tuple => tuple.Score); // Order memories by their score

            // Pick the memory with the highest score as the representative for the cluster
            representativeMemories.Add(weightedMemories.First().Memory);
        }

        return representativeMemories.Distinct();
    }

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    public async Task AddMemoryAsync(T newMemoryContent)
    {
        // Recall memories related to this new content
        var rawMemoryEmbedding = await GenerateEmbeddingAsync(newMemoryContent);

        AdjustBreadthAndDepthBasedOnRelevance(rawMemoryEmbedding);

        var recollections = WeightedMemoryRecall(rawMemoryEmbedding).Take(ThoughtBreadth);

        var newMemory = new CortexMemory<T>(newMemoryContent, rawMemoryEmbedding)
        {
            CreationTimestamp = DateTime.Now,
            Type = CortexMemoryType.Core,
        };

        Memories.Add(newMemory);

        // Remember the act of recalling these memories, and roll reflections from one recollection to the next.
        var allRelevantMemories = new Dictionary<DateTime, CortexMemory<T>>(recollections.ToDictionary(x => x.CreationTimestamp));

        // Memory loop.
        // Roughly emulates the act of remembering and reflecting on thoughts before responding.
        // Context is rolled from the original memory, through a timeline of the most relevant and recent memories, and out into a "final thought".
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

        var recollectionMemory = await SummarizeMemoryInNewContext(newMemory, relevantMemories);
        var recollectionMemoryEmbedding = await GenerateEmbeddingAsync(recollectionMemory);

        var memoryOfRecollection = new CortexMemory<T>(recollectionMemory, recollectionMemoryEmbedding)
        {
            CreationTimestamp = DateTime.Now,
            Type = CortexMemoryType.RecalledWithContext,
        };

        allRelevantMemories.Add(memoryOfRecollection.CreationTimestamp, memoryOfRecollection);
        Memories.Add(memoryOfRecollection);

        // Create final reaction to the new memory, but with recent internal reflections.
        // Recency weightes ensure recent recollections are prioritized over old ones.
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

    private void AdjustBreadthAndDepthBasedOnRelevance(double[] embedding)
    {
        var topMemories = WeightedMemoryRecall(embedding).ToList();

        // If no memories are available, return without adjusting
        if (!topMemories.Any())
            return;

        var averageRelevance = topMemories.Average(memory => ComputeCosineSimilarity(embedding, memory.EmbeddingVector));

        if (averageRelevance > 0.8)  // High relevance threshold
        {
            if (AutoAdjustThoughtBreadth)
                ThoughtBreadth = Math.Min(ThoughtBreadth + 5, ThoughtBreadthMax);  // Max limit

            if (AutoAdjustThoughtDepth)
                ThoughtDepth = Math.Min(ThoughtDepth + 1, ThoughtDepthMax);      // Max limit
        }
        else if (averageRelevance < 0.5)  // Low relevance threshold
        {
            if (AutoAdjustThoughtBreadth)
                ThoughtBreadth = Math.Max(ThoughtBreadth - 1, 1);  // Min limit

            if (AutoAdjustThoughtDepth)
                ThoughtDepth = Math.Max(ThoughtDepth - 1, 1);      // Min limit
        }
    }

    /// <summary>
    /// Clusters similar memories together based on their embedding similarity.
    /// Within each cluster, memories are weighted by their recency.
    /// </summary>
    /// <param name="memories">The collection of memories to be clustered.</param>
    /// <param name="similarityThreshold">The threshold above which memories are considered similar and clustered together.</param>
    /// <returns>A sequence of clusters where each cluster is a sequence of similar memories.</returns>
    public IEnumerable<IEnumerable<CortexMemory<T>>> ClusterSimilarMemories(IEnumerable<CortexMemory<T>> memories, double similarityThreshold)
    {
        var clusters = new List<List<CortexMemory<T>>>();

        // Iterate through each memory to determine its cluster
        foreach (var memory in memories)
        {
            bool foundCluster = false;

            // For each existing cluster, compute its weighted average embedding
            // and check the similarity with the current memory.
            foreach (var cluster in clusters)
            {
                var clusterAverage = ComputeWeightedAverageEmbedding(cluster);

                // If the cluster is empty, skip to the next cluster
                if (clusterAverage is null)
                    continue;

                // If the cosine similarity of the current memory and the cluster's average embedding
                // exceeds the threshold, the memory is added to that cluster.
                if (ComputeCosineSimilarity(memory.EmbeddingVector, clusterAverage) > similarityThreshold)
                {
                    cluster.Add(memory);
                    foundCluster = true;
                    break;
                }
            }

            // If the memory does not belong to any existing cluster, create a new cluster for it.
            if (!foundCluster)
            {
                clusters.Add(new List<CortexMemory<T>> { memory });
            }
        }

        return clusters;
    }

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

}
