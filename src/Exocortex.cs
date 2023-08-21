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
public abstract class Exocortex<T>
{
    /// <summary>
    /// All memories created by the agent, in the order they were created.
    /// </summary>
    public SortedSet<CortexMemory<T>> Memories { get; } = new SortedSet<CortexMemory<T>>();

    /// <summary>
    /// Gets or sets the number of past memories to reframe in light of a new memory.
    /// </summary>
    public int ThoughtDepth { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of related memories to recall for context when reframing a memory.
    /// </summary>
    public int ThoughtBreadth { get; set; } = 6;

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
    public int ThoughtDepthMax { get; set; } = 6;

    /// <summary>
    /// The maximum Breadth of <see cref="ThoughtBreadth"/>.
    /// </summary>
    public int ThoughtBreadthMax { get; set; } = 6;

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
    /// Computes the cosine similarity between two vectors.
    /// This can be used to determine how similar two memories are.
    /// </summary>
    public double ComputeCosineSimilarity(double[] vector1, double[] vector2)
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

        // Parameters for the logistic function
        double x0 = 1;  // Midpoint of the curve (in minutes)
        double k = 0.1;  // Steepness factor

        // Logistic decay function
        var recencyScore = 1 / (1 + Math.Exp(k * (minutesSinceCreation - x0)));

        return recencyScore;
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

        var recollections = WeightedMemoryRecall(rawMemoryEmbedding);
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
        foreach (var memory in recollections.OrderBy(x => x.CreationTimestamp).Take(ThoughtBreadth))
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
            .Take(25)
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
        var reaction = await ReactToMemoryAsync(newMemory, allRelevantMemories.Values.Take(25).OrderBy(x => x.CreationTimestamp));
        var reactionEmbedding = await GenerateEmbeddingAsync(reaction);

        var reactionMemory = new CortexMemory<T>(reaction, reactionEmbedding)
        {
            CreationTimestamp = DateTime.Now,
            Type = CortexMemoryType.Reaction,
        };

        Memories.Add(reactionMemory);
    }

    /// <summary>
    /// Retrieves and ranks memories relevant to a given query content.
    /// </summary>
    /// <param name="embedding">The embeddings for this content, if available.</param>
    /// <returns>An ordered set of memories, ranked by relevance, importance, and recency.</returns>
    public IEnumerable<CortexMemory<T>> WeightedMemoryRecall(double[] embedding)
    {
        return Memories
            .Select(memory =>
            {
                var relevance = ComputeCosineSimilarity(embedding, memory.EmbeddingVector);
                var recency = ComputeRecencyScore(memory.CreationTimestamp);

                return (Memory: memory, Score: relevance * recency * recency);
            })
            .OrderByDescending(tuple => tuple.Score)
            .ThenByDescending(tuple => tuple.Memory.CreationTimestamp)
            .Select(tuple => tuple.Memory);
    }

    public void AdjustBreadthAndDepthBasedOnRelevance(double[] embedding)
    {
        var topMemories = WeightedMemoryRecall(embedding).ToList();  // Take the top 10 memories

        // If no memories are available, return without adjusting
        if (!topMemories.Any())
            return;

        var averageRelevance = topMemories.Average(memory => ComputeCosineSimilarity(embedding, memory.EmbeddingVector));

        if (averageRelevance > 0.8)  // High relevance threshold
        {
            if (AutoAdjustThoughtBreadth)
                ThoughtBreadth = Math.Min(ThoughtBreadth + 1, ThoughtBreadthMax);  // Max limit

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
}
