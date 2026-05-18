using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OwlCore.AI.Exocortex.AppModels;

namespace OwlCore.AI.Exocortex.DataModels;

/// <summary>
/// Maps between Exocortex application models, data records, and current engine memory records.
/// </summary>
public static class CortexMemoryDataMapper
{
    /// <summary>
    /// Creates an application memory model from a persistence-friendly data record.
    /// </summary>
    public static ICortexMemory<T> ToAppModel<T>(CortexMemoryData<T> data, Func<string, CancellationToken, Task<ICortexMemory<T>?>>? resolveMemoryAsync = null)
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));

        if (data.Type == CortexMemoryType.Recollection && data.RecalledMemoryIds.Count > 0)
        {
            return new RecollectionCortexMemoryAppModel<T>(
                data.Id,
                data.Content,
                data.EmbeddingVectors,
                data.CreationTimestamp,
                data.RecalledMemoryIds,
                resolveMemoryAsync ?? ResolveNoMemoryAsync<T>);
        }

        return new CortexMemoryAppModel<T>(data.Id, data.Type, data.Content, data.EmbeddingVectors, data.CreationTimestamp);
    }

    /// <summary>
    /// Creates a persistence-friendly data record from an application memory model.
    /// </summary>
    public static async Task<CortexMemoryData<T>> ToDataAsync<T>(ICortexMemory<T> memory, CancellationToken cancellationToken = default)
    {
        if (memory is null)
            throw new ArgumentNullException(nameof(memory));

        var recalledMemoryIds = new List<string>();
        if (memory is IRecollectionCortexMemory<T> recollection)
        {
            await foreach (var recalledMemory in recollection.GetRecalledMemoriesAsync(cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                recalledMemoryIds.Add(recalledMemory.Id);
            }
        }

        return new CortexMemoryData<T>
        {
            Id = memory.Id,
            Type = memory.Type,
            Content = memory.Content,
            EmbeddingVectors = memory.EmbeddingVectors.ToArray(),
            CreationTimestamp = memory.CreationTimestamp,
            RecalledMemoryIds = recalledMemoryIds,
        };
    }

    /// <summary>
    /// Creates a persistence-friendly data record from a current engine memory record.
    /// </summary>
    public static CortexMemoryData<T> ToData<T>(string id, CortexMemory<T> memory, Func<CortexMemory<T>, string?>? getMemoryId = null)
    {
        if (id is null)
            throw new ArgumentNullException(nameof(id));
        if (memory is null)
            throw new ArgumentNullException(nameof(memory));

        var recalledMemoryIds = new List<string>();
        if (memory is RecollectionCortexMemory<T> recollection && getMemoryId is not null)
        {
            foreach (var recalledMemory in recollection.RecalledMemories)
            {
                var recalledMemoryId = getMemoryId(recalledMemory);
                if (!string.IsNullOrWhiteSpace(recalledMemoryId))
                    recalledMemoryIds.Add(recalledMemoryId!);
            }
        }

        return new CortexMemoryData<T>
        {
            Id = id,
            Type = memory.Type,
            Content = memory.Content,
            EmbeddingVectors = memory.EmbeddingVectors.ToArray(),
            CreationTimestamp = memory.CreationTimestamp,
            RecalledMemoryIds = recalledMemoryIds,
        };
    }

    /// <summary>
    /// Creates a current engine memory record from a persistence-friendly data record.
    /// </summary>
    public static CortexMemory<T> ToCortexMemory<T>(CortexMemoryData<T> data, IReadOnlyDictionary<string, CortexMemory<T>>? memoriesById = null)
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));

        if (data.Type == CortexMemoryType.Recollection && data.RecalledMemoryIds.Count > 0 && memoriesById is not null)
        {
            var recalledMemories = data.RecalledMemoryIds
                .Select(memoryId => memoriesById.TryGetValue(memoryId, out var memory) ? memory : null)
                .Where(memory => memory is not null)
                .Cast<CortexMemory<T>>()
                .ToArray();

            return new RecollectionCortexMemory<T>(data.Content, data.EmbeddingVectors.ToArray(), recalledMemories, data.CreationTimestamp);
        }

        return new CortexMemory<T>(data.Content, data.EmbeddingVectors.ToArray(), data.CreationTimestamp)
        {
            Type = data.Type,
        };
    }

    private static Task<ICortexMemory<T>?> ResolveNoMemoryAsync<T>(string memoryId, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult<ICortexMemory<T>?>(null);
    }
}
