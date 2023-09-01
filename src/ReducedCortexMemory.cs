namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents a memory with reduced embeddings.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record ReducedCortexMemory<T> : CortexMemory<T>
{
    /// <summary>
    /// Creates a new instance of <see cref="RecollectionCortexMemory{T}"/>.
    /// </summary>
    /// <param name="reducedEmbeddingVectors">The vectorized embeddings that represent this memory.</param>
    /// <param name="originalMemory">The original memory was reduced to create this.</param>
    public ReducedCortexMemory(double[] reducedEmbeddingVectors, CortexMemory<T> originalMemory)
        : base(originalMemory.Content, reducedEmbeddingVectors)
    {
        Type = CortexMemoryType.Recollection;
        OriginalMemory = originalMemory;
    }

    /// <summary>
    /// The memories that were recalled to create this memory.
    /// </summary>
    public CortexMemory<T> OriginalMemory { get; }
    
    /// <inheritdoc/>
    public override string ToString()
    {
        return $"{CreationTimestamp}: {Content}";
    }
}
