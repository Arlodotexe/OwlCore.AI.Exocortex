using System;
using System.Collections.Generic;
using OwlCore.AI.Exocortex.DataModels;

namespace OwlCore.AI.Exocortex.AppModels;

/// <summary>
/// Default application-facing memory implementation.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public class CortexMemoryAppModel<T> : ICortexMemory<T>
{
    /// <summary>
    /// Creates a new instance of <see cref="CortexMemoryAppModel{T}"/>.
    /// </summary>
    public CortexMemoryAppModel(CortexMemoryData<T> data)
    {
        Data = data ?? throw new ArgumentNullException(nameof(data));
    }

    /// <summary>
    /// The underlying data model for this application model.
    /// </summary>
    public CortexMemoryData<T> Data { get; }

    /// <inheritdoc/>
    public string Id => Data.Id;

    /// <inheritdoc/>
    public CortexMemoryType Type => Data.Type;

    /// <inheritdoc/>
    public T Content => Data.Content;

    /// <inheritdoc/>
    public DateTime CreationTimestamp => Data.CreationTimestamp;

    /// <inheritdoc/>
    public IReadOnlyList<float> EmbeddingVectors => Data.EmbeddingVectors;

    /// <summary>
    /// Creates a copy of the embedding vector for mutable consumers.
    /// </summary>
    public float[] GetEmbeddingVectorCopy()
    {
        return (float[])Data.EmbeddingVectors.Clone();
    }
}
