using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents the Exocortex - a memory processing unit.
/// </summary>
/// <typeparam name="T">The type of raw content the memories hold.</typeparam>
public abstract class Exocortex<T>
{
    private List<Memory<T>> _memories = new List<Memory<T>>();

    /// <summary>
    /// Gets or sets the decay factor for memory recency computations.
    /// Used to model how quickly a memory fades over time.
    /// </summary>
    public double MemoryDecayFactor { get; private set; } = 0.995;

    /// <summary>
    /// Defines how the Exocortex should react to new content and related memories.
    /// </summary>
    public abstract Task<T> ExperienceTickAsync(T newContent, IEnumerable<Memory<T>> relatedMemories, double importanceToRelatedMemories);

    /// <summary>
    /// Generates an embedding vector for a given memory content.
    /// Used for computing similarities between memories.
    /// </summary>
    /// <param name="memoryContent">The content to generate an embedding for.</param>
    /// <returns>A vector representing the content.</returns>
    public abstract double[] GenerateEmbedding(T memoryContent);

    /// <summary>
    /// Generates an importance score for a given memory content.
    /// This score reflects how critical or poignant the memory is.
    /// </summary>
    /// <param name="memoryContent">The content to generate an importance score for.</param>
    /// <param name="relatedMemories">The memories related to the given content.</param>
    /// <returns>The calculated importance score.</returns>
    public abstract Task<double> GenerateImportanceScore(T memoryContent, IEnumerable<Memory<T>> relatedMemories);

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
        var recencyScore = Math.Pow(MemoryDecayFactor, timeSpanSinceCreation.TotalHours);

        return recencyScore;
    }

    /// <summary>
    /// Adds a new memory to the Exocortex, turning objective experiences into subjective experiences.
    /// </summary>
    /// <param name="newMemoryContent">The content of the new memory.</param>
    public async Task AddMemoryAsync(T newMemoryContent)
    {
        // Recall memories related to this new content
        var rawMemoryEmbedding = GenerateEmbedding(newMemoryContent);
        var recollections = WeightedMemoryRecall(rawMemoryEmbedding);
        var objectiveNewMemoryImportance = await GenerateImportanceScore(newMemoryContent, recollections);

        // Interpret raw objective experience + related recollections + calculated objective importance
        var subjectiveTick = await ExperienceTickAsync(newMemoryContent, recollections, objectiveNewMemoryImportance);
        var subjectiveEmbedding = GenerateEmbedding(subjectiveTick);

        // Extract relevance distance between subjective and objective experience.
        var currentRelevance = ComputeCosineSimilarity(subjectiveEmbedding, rawMemoryEmbedding);

        // Create new memory with the subject interpretation of objective experience.
        var subjectiveMemory = new Memory<T>(newMemoryContent, rawMemoryEmbedding, objectiveNewMemoryImportance, currentRelevance);

        // Store subjective memory
        _memories.Add(subjectiveMemory);

        // Find memories related to the subjective version of this memory.
        var subjectiveRecollections = WeightedMemoryRecall(subjectiveMemory.EmbeddingVector);

        // Remember the act of recalling these memories subjectively
        foreach (var memory in recollections)
        {
            // How important is this recalled memory in light of the new memory?
            var subjectiveRecalledMemoryImportance = await GenerateImportanceScore(memory.Content, subjectiveRecollections);

            // Interpret past memory + recollections about the new memory + newly calculated subjective importance.
            var reframedContent = await ExperienceTickAsync(memory.Content, subjectiveRecollections, importanceToRelatedMemories: subjectiveRecalledMemoryImportance);
            var reframedEmbedding = GenerateEmbedding(reframedContent);

            var reframedMemory = new Memory<T>(reframedContent, reframedEmbedding, subjectiveRecalledMemoryImportance, currentRelevance)
            {
                CreationTimestamp = DateTime.Now,
            };

            _memories.Add(reframedMemory);
        }

        // Consolidate into a "fully formed thought" in light of the (now) subjectively re-evaluated related memories
        // Recency weightes ensure recent recollections are prioritized over old ones.
        // Relevance weights ensure we can filter through large volumes of incoming information.
        var insightGuidedRecollections = WeightedMemoryRecall(subjectiveMemory.EmbeddingVector);

        // How subjectively important is this new memory in light of new insights?
        var insightGuidedSubjectiveMemoryImportance = await GenerateImportanceScore(subjectiveMemory.Content, insightGuidedRecollections);

        // Create final memory based on previous subjective experience.
        var insightMemoryContent = await ExperienceTickAsync(subjectiveMemory.Content, subjectiveRecollections, insightGuidedSubjectiveMemoryImportance);
        var insightMemoryEmbedding = GenerateEmbedding(insightMemoryContent);
        var insightMemoryImportance = await GenerateImportanceScore(insightMemoryContent, insightGuidedRecollections);

        var consolidatedMemory = new Memory<T>(insightMemoryContent, insightMemoryEmbedding, insightMemoryImportance, currentRelevance)
        {
            CreationTimestamp = DateTime.Now,
        };

        // Store the consolidated memory
        _memories.Add(consolidatedMemory);
    }

    /// <summary>
    /// Retrieves and ranks memories relevant to a given query content.
    /// </summary>
    /// <param name="embedding">The embeddings for this content, if available.</param>
    /// <returns>An ordered set of memories, ranked by relevance, importance, and recency.</returns>
    public IEnumerable<Memory<T>> WeightedMemoryRecall(double[] embedding)
    {
        return _memories.Select(memory =>
        {
            var relevance = ComputeCosineSimilarity(embedding, memory.EmbeddingVector);
            var recency = ComputeRecencyScore(memory.CreationTimestamp);

            return (Memory: memory, Score: relevance * memory.ImportanceOnCreation * recency);
        })
        .OrderByDescending(tuple => tuple.Score)
        .Select(tuple => tuple.Memory);
    }
}
