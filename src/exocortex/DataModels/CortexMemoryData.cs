using System;
using System.Collections.Generic;

namespace OwlCore.AI.Exocortex.DataModels;

/// <summary>
/// Persistence-friendly memory data record.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public record CortexMemoryData<T>
{
    /// <summary>
    /// Gets the memory identifier within the current storage boundary.
    /// </summary>
    public string Id { get; init; } = string.Empty;

    /// <summary>
    /// Gets the memory type.
    /// </summary>
    public CortexMemoryType Type { get; init; }

    /// <summary>
    /// Gets the memory content.
    /// </summary>
    public T Content { get; init; } = default!;

    /// <summary>
    /// Gets the embedding vector representing the memory.
    /// </summary>
    public float[] EmbeddingVectors { get; init; } = Array.Empty<float>();

    /// <summary>
    /// Gets the timestamp of when this memory was created.
    /// </summary>
    public DateTime CreationTimestamp { get; init; }

    /// <summary>
    /// Gets memory ids recalled to create this memory.
    /// </summary>
    public IReadOnlyList<string> RecalledMemoryIds { get; init; } = Array.Empty<string>();
}
