using System;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// A memory that was weighted against the short-term. If the score is high enough, it qualifies to be re-inserted into short-term memory.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record WorkingCortexMemory<T> : CortexMemory<T>
{
    /// <summary>
    /// Creates a new instance of <see cref="RecollectionCortexMemory{T}"/>.
    /// </summary>
    /// <param name="originalMemory">The original memory was reduced to create this.</param>
    /// <param name="score">The score for this memory, if any.</param>
    public WorkingCortexMemory(CortexMemory<T> originalMemory, double score)
        : base(originalMemory.Content, originalMemory.EmbeddingVectors, originalMemory.CreationTimestamp)
    {
        Type = CortexMemoryType.Recollection;
        WeighedMemory = originalMemory;
        Score = score;
    }

    /// <summary>
    /// The memory that was weighed against the short-term.
    /// </summary>
    public CortexMemory<T> WeighedMemory { get; }

    /// <summary>
    /// The score for this memory, if any.
    /// </summary>
    public double Score { get; set; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"{CreationTimestamp}: {Content}";
    }
}
