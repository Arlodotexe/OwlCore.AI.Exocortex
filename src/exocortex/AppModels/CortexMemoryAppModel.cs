using System;
using System.Collections.Generic;

namespace OwlCore.AI.Exocortex.AppModels;

/// <summary>
/// Default application-facing memory implementation.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public class CortexMemoryAppModel<T> : ICortexMemory<T>
{
    private readonly float[] _embeddingVectors;

    /// <summary>
    /// Creates a new instance of <see cref="CortexMemoryAppModel{T}"/>.
    /// </summary>
    public CortexMemoryAppModel(string id, CortexMemoryType type, T content, IEnumerable<float> embeddingVectors, DateTime creationTimestamp)
    {
        if (id is null)
            throw new ArgumentNullException(nameof(id));
        if (embeddingVectors is null)
            throw new ArgumentNullException(nameof(embeddingVectors));

        Id = id;
        Type = type;
        Content = content;
        _embeddingVectors = ToArray(embeddingVectors);
        CreationTimestamp = creationTimestamp;
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public CortexMemoryType Type { get; }

    /// <inheritdoc/>
    public T Content { get; }

    /// <inheritdoc/>
    public DateTime CreationTimestamp { get; }

    /// <inheritdoc/>
    public IReadOnlyList<float> EmbeddingVectors => _embeddingVectors;

    /// <summary>
    /// Creates a copy of the embedding vector for mutable consumers.
    /// </summary>
    public float[] GetEmbeddingVectorCopy()
    {
        return (float[])_embeddingVectors.Clone();
    }

    private static float[] ToArray(IEnumerable<float> values)
    {
        if (values is float[] array)
            return (float[])array.Clone();

        return System.Linq.Enumerable.ToArray(values);
    }
}
