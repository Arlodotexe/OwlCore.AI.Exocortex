using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.Tests;

internal sealed class RecordingTextExocortex : Exocortex<string>
{
    public RecordingTextExocortex()
    {
        Embeddings["query"] = [1f, 0f, 0f];
        Embeddings["similar"] = [1f, 0f, 0f];
        Embeddings["old-similar"] = [1f, 0f, 0f];
        Embeddings["seed"] = [1f, 0f, 0f];
        Embeddings["orthogonal"] = [0f, 1f, 0f];
        Embeddings["old-orthogonal"] = [0f, 1f, 0f];
        Embeddings["recent-unrelated"] = [0f, 1f, 0f];
        Embeddings["heartbeat"] = [0f, 1f, 0f];
        Embeddings["opposite"] = [-1f, 0f, 0f];
        Embeddings["old-opposite"] = [-1f, 0f, 0f];
    }

    public Dictionary<string, float[]> Embeddings { get; } = new(StringComparer.Ordinal);

    public List<CortexMemory<string>> LastReactionContext { get; private set; } = [];

    public List<List<CortexMemory<string>>> ReactionContexts { get; } = [];

    public CortexMemory<string> AddExisting(string content, DateTime creationTimestamp, CortexMemoryType type = CortexMemoryType.Core)
    {
        var memory = new CortexMemory<string>(content, EmbeddingFor(content), creationTimestamp)
        {
            Type = type,
        };

        Memories.Add(memory);
        return memory;
    }

    public float[] EmbeddingFor(string content)
    {
        return Embeddings.TryGetValue(content, out var embedding) ? embedding : [0f, 1f, 0f];
    }

    public override Task<string> SummarizeMemoryInNewContext(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
    {
        return Task.FromResult($"summary: {memory.Content}");
    }

    public override Task<string> ReactToMemoryAsync(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
    {
        LastReactionContext = workingMemories.ToList();
        ReactionContexts.Add(LastReactionContext);
        return Task.FromResult($"reaction: {memory.Content}");
    }

    public override Task<float[]> GenerateEmbeddingAsync(string memoryContent, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(EmbeddingFor(memoryContent));
    }
}
