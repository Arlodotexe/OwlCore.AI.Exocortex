using ClusterF_ck.Spaces.Properties;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Computes the memory weights for use in clustering.
/// </summary>
/// <typeparam name="T">The raw memory type.</typeparam>
public struct CortexMemoryDistanceSpace<T> : IDistanceSpace<CortexMemory<T>>
{
    private readonly Exocortex<T> _exocortex;

    /// <summary>
    /// Creates a new instance of <see cref="CortexMemoryDistanceSpace{T}"/>.
    /// </summary>
    /// <param name="exocortex">The exocortex used to compute the memory weights </param>
    public CortexMemoryDistanceSpace(Exocortex<T> exocortex) => _exocortex = exocortex;

    /// <inheritdoc/>
    public double FindDistanceSquared(CortexMemory<T> it1, CortexMemory<T> it2) => _exocortex.ComputeMemoryWeight(it1, it2.EmbeddingVector);
}
