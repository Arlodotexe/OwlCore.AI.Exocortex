using System;
using System.Collections.Generic;

namespace OwlCore.AI.Exocortex;


/// <summary>
/// Represents a memory composed of the recollection of other memories.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record RecollectionCortexMemory<T> : CortexMemory<T>
{
    /// <summary>
    /// Creates a new instance of <see cref="RecollectionCortexMemory{T}"/>.
    /// </summary>
    /// <param name="content">The raw content of the memory.</param>
    /// <param name="embeddingVector">The vectorized embeddings that represent this memory.</param>
    /// <param name="recalledMemories">The memories that were recalled to create this.</param>
    /// <param name="creationTimestamp">Gets the timestamp of when this memory was created.</param>
    public RecollectionCortexMemory(T content, float[] embeddingVector, IEnumerable<CortexMemory<T>> recalledMemories, DateTime creationTimestamp)
        : base(content, embeddingVector, creationTimestamp)
    {
        Type = CortexMemoryType.Recollection;
        RecalledMemories = recalledMemories;
    }

    /// <summary>
    /// The memories that were recalled to create this memory.
    /// </summary>
    public IEnumerable<CortexMemory<T>> RecalledMemories { get; }
    
    /// <inheritdoc/>
    public override string ToString()
    {
        return $"{CreationTimestamp}: {Content}";
    }
}
