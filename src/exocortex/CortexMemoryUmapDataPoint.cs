using UMAP;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Represents a data point that can be operated on in Umap calculations.
/// </summary>
/// <typeparam name="T"></typeparam>
public record CortexMemoryUmapDataPoint<T> : IUmapDataPoint
{
    /// <summary>
    /// Creates a new instance of <see cref="CortexMemoryUmapDataPoint{T}"/>.
    /// </summary>
    /// <param name="memory"></param>
    public CortexMemoryUmapDataPoint(CortexMemory<T> memory)
    {
        Memory = memory;
    }

    /// <summary>
    /// The original memory object.
    /// </summary>
    public CortexMemory<T> Memory { get; }

    /// <inheritdoc/>
    public float[] Data => Memory.EmbeddingVectors;

    /// <summary>
    /// Implicit conversation back to <see cref="CortexMemory{T}"/>.
    /// </summary>
    /// <param name="x"></param>
    public static implicit operator CortexMemory<T>(CortexMemoryUmapDataPoint<T> x) => x.Memory;
}
