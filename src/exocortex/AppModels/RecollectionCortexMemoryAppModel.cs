using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.AppModels;

/// <summary>
/// Default application-facing recollection memory implementation.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public class RecollectionCortexMemoryAppModel<T> : CortexMemoryAppModel<T>, IRecollectionCortexMemory<T>
{
    private readonly Func<string, CancellationToken, Task<ICortexMemory<T>?>> _resolveMemoryAsync;

    /// <summary>
    /// Creates a new instance of <see cref="RecollectionCortexMemoryAppModel{T}"/>.
    /// </summary>
    public RecollectionCortexMemoryAppModel(string id, T content, IEnumerable<float> embeddingVectors, DateTime creationTimestamp, IReadOnlyList<string> recalledMemoryIds, Func<string, CancellationToken, Task<ICortexMemory<T>?>> resolveMemoryAsync)
        : base(id, CortexMemoryType.Recollection, content, embeddingVectors, creationTimestamp)
    {
        if (recalledMemoryIds is null)
            throw new ArgumentNullException(nameof(recalledMemoryIds));
        if (resolveMemoryAsync is null)
            throw new ArgumentNullException(nameof(resolveMemoryAsync));

        RecalledMemoryIds = System.Linq.Enumerable.ToArray(recalledMemoryIds);
        _resolveMemoryAsync = resolveMemoryAsync;
    }

    internal IReadOnlyList<string> RecalledMemoryIds { get; }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ICortexMemory<T>> GetRecalledMemoriesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var recalledMemoryId in RecalledMemoryIds)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var memory = await _resolveMemoryAsync(recalledMemoryId, cancellationToken).ConfigureAwait(false);
            if (memory is not null)
                yield return memory;
        }
    }
}
