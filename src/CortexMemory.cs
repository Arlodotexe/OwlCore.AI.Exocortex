using System;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents a single memory in the exocortex, with content and associated metadata.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record CortexMemory<T> : IComparable<CortexMemory<T>>
{
    /// <summary>
    /// Creates a new instance of <see cref="CortexMemory{T}"/>.
    /// </summary>
    /// <param name="content">The raw content of the memory.</param>
    /// <param name="embeddingVector">The vectorized embeddings that represent this memory.</param>
    public CortexMemory(T content, double[] embeddingVector)
    {
        Content = content;
        EmbeddingVector = embeddingVector;
    }

    /// <summary>
    /// The memory type.
    /// </summary>
    public CortexMemoryType Type { get; set; }

    /// <summary>
    /// Gets the content of the memory.
    /// </summary>
    public T Content { get; init; }

    /// <summary>
    /// Gets the timestamp of when this memory was created.
    /// </summary>
    public DateTime CreationTimestamp { get; init; } = DateTime.Now;

    /// <summary>
    /// Gets the embedding vector representing the content of this memory.
    /// </summary>
    public double[] EmbeddingVector { get; init; }

    /// <inheritdoc/>
    public int CompareTo(CortexMemory<T>? other)
    {
        return CreationTimestamp.CompareTo(other?.CreationTimestamp);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"[{CreationTimestamp}] {Content}";
    }
}
