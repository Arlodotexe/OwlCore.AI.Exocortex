using System;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents a single memory in the exocortex, with content and associated metadata.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record Memory<T>
{
    /// <summary>
    /// Represents a single memory in the exocortex, with content and associated metadata.
    /// </summary>
    /// <param name="content">The raw content of the memory.</param>
    /// <param name="embeddingVector">The vectorized embeddings that represent this memory.</param>
    /// <param name="currentImportance">The importance of this memory temporal continuity to when it was created.</param>
    /// <param name="currentRelevance">The relevance of this memory with temporal continuity to when it was created.</param>
    public Memory(T content, double[] embeddingVector, double currentImportance, double currentRelevance)
    {
        Content = content;
        EmbeddingVector = embeddingVector;
        ImportanceOnCreation = currentImportance;
        RelevanceOnCreation = currentRelevance;
    }

    /// <summary>
    /// Gets the content of the memory.
    /// </summary>
    public T Content { get; init; }

    /// <summary>
    /// Gets the timestamp of when this memory was created.
    /// </summary>
    public DateTime CreationTimestamp { get; init; } = DateTime.Now;

    /// <summary>
    /// Gets or sets the importance of this memory when it was created.
    /// </summary>
    public double ImportanceOnCreation { get; init; }

    /// <summary>
    /// Gets or sets the relevance of this memory when it was created.
    /// </summary>
    public double RelevanceOnCreation { get; set; }

    /// <summary>
    /// Gets the embedding vector representing the content of this memory.
    /// </summary>
    public double[] EmbeddingVector { get; init; }
}
