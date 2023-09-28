using CommunityToolkit.Diagnostics;
using HdbscanSharp.Distance;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Computes the memory weights for use in clustering.
/// </summary>
/// <typeparam name="T">The raw memory type.</typeparam>
public struct CortexMemoryDistanceSpace<T> : IDistanceCalculator<CortexMemory<T>>
{
    private readonly Exocortex<T> _exocortex;

    /// <summary>
    /// Creates a new instance of <see cref="CortexMemoryDistanceSpace{T}"/>.
    /// </summary>
    /// <param name="exocortex"></param>
    public CortexMemoryDistanceSpace(Exocortex<T> exocortex)
    {
        Guard.IsNotNull(exocortex);
        _exocortex = exocortex;
    }

    /// <inheritdoc/>
    public double ComputeDistance(int indexOne, int indexTwo, CortexMemory<T> attributesOne, CortexMemory<T> attributesTwo) => _exocortex.ComputeFullMemoryWeight(attributesOne, attributesTwo.EmbeddingVectors);
}
