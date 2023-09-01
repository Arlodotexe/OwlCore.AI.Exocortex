using HdbscanSharp.Distance;

namespace OwlCore.AI.Exocortex;

/// <summary>
/// Computes the memory weights for use in clustering.
/// </summary>
/// <typeparam name="T">The raw memory type.</typeparam>
public struct CortexMemoryDistanceSpace<T> : IDistanceCalculator<CortexMemory<T>>
{
    public double ComputeDistance(int indexOne, int indexTwo, CortexMemory<T> attributesOne, CortexMemory<T> attributesTwo) => Exocortex<T>.ComputeCosineSimilarity(attributesOne.EmbeddingVector, attributesTwo.EmbeddingVector);
}
