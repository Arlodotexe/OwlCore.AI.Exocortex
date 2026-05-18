using System.Collections.Generic;
using System.Threading;

namespace OwlCore.AI.Exocortex.AppModels;

/// <summary>
/// Application-facing memory model for a memory formed by recalling other memories.
/// </summary>
/// <typeparam name="T">The type of content this memory holds.</typeparam>
public interface IRecollectionCortexMemory<T> : ICortexMemory<T>
{
    /// <summary>
    /// Resolves the memories recalled to create this memory.
    /// </summary>
    /// <param name="cancellationToken">A token that can be used to cancel the ongoing operation.</param>
    IAsyncEnumerable<ICortexMemory<T>> GetRecalledMemoriesAsync(CancellationToken cancellationToken = default);
}
