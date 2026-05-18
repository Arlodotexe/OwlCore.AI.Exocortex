using System;
using System.Collections.Generic;

namespace OwlCore.AI.Exocortex.AppModels;

/// <summary>
/// Application-facing memory model used by Exocortex layers above storage and transport records.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public interface ICortexMemory<T>
{
    /// <summary>
    /// Gets the memory identifier within the current application model boundary.
    /// </summary>
    string Id { get; }

    /// <summary>
    /// Gets the memory type.
    /// </summary>
    CortexMemoryType Type { get; }

    /// <summary>
    /// Gets the memory content.
    /// </summary>
    T Content { get; }

    /// <summary>
    /// Gets the timestamp of when this memory was created.
    /// </summary>
    DateTime CreationTimestamp { get; }

    /// <summary>
    /// Gets the embedding vector representing this memory.
    /// </summary>
    IReadOnlyList<float> EmbeddingVectors { get; }
}
