using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OwlCore.AI.Exocortex.DataModels;

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
    public RecollectionCortexMemoryAppModel(CortexMemoryData<T> data, Func<string, CancellationToken, Task<ICortexMemory<T>?>> resolveMemoryAsync)
        : base(data)
    {
        if (data.Type != CortexMemoryType.Recollection)
            throw new ArgumentException("Recollection app models require recollection data.", nameof(data));

        _resolveMemoryAsync = resolveMemoryAsync ?? throw new ArgumentNullException(nameof(resolveMemoryAsync));
    }

    /// <summary>
    /// Gets memory ids recalled to create this memory.
    /// </summary>
    public IReadOnlyList<string> RecalledMemoryIds => Data.RecalledMemoryIds;

    /// <inheritdoc/>
    public async IAsyncEnumerable<ICortexMemory<T>> GetRecalledMemoriesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var recalledMemoryId in Data.RecalledMemoryIds)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var memory = await _resolveMemoryAsync(recalledMemoryId, cancellationToken).ConfigureAwait(false);
            if (memory is not null)
                yield return memory;
        }
    }
}
