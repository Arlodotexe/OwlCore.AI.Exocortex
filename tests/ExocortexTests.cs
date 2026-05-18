namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class ExocortexTests
{
    private static readonly DateTime Present = new(2026, 5, 17, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public void ComputeCosineSimilarityReturnsOneForIdenticalVectors()
    {
        var similarity = Exocortex<string>.ComputeCosineSimilarity([1f, 2f, 3f], [1f, 2f, 3f]);

        Assert.AreEqual(1f, similarity, 0.0001f);
    }

    [TestMethod]
    public void ComputeCosineSimilarityRejectsMismatchedVectorDimensions()
    {
        Assert.ThrowsException<ArgumentException>(() => Exocortex<string>.ComputeCosineSimilarity([1f, 2f], [1f]));
    }

    [TestMethod]
    public void ComputeCosineSimilarityRejectsZeroMagnitudeVectors()
    {
        Assert.ThrowsException<ArgumentException>(() => Exocortex<string>.ComputeCosineSimilarity([0f, 0f], [1f, 0f]));
    }

    [TestMethod]
    public void ComputeRecencyWeightDecreasesAcrossLongTermAge()
    {
        var exocortex = new TestExocortex
        {
            CustomPresentDateTime = Present,
            ShortTermMemoryDuration = TimeSpan.FromHours(8),
            LongTermDecayThreshold = 0.01f,
        };

        exocortex.Memories.Add(new CortexMemory<string>("newer", TestExocortex.EmbeddingFor("alpha"), Present.AddHours(-24)) { Type = CortexMemoryType.Core });
        exocortex.Memories.Add(new CortexMemory<string>("older", TestExocortex.EmbeddingFor("alpha"), Present.AddHours(-48)) { Type = CortexMemoryType.Core });

        var newerWeight = exocortex.ComputeRecencyWeight(Present.AddHours(-24));
        var olderWeight = exocortex.ComputeRecencyWeight(Present.AddHours(-48));

        Assert.IsTrue(newerWeight > olderWeight, $"Expected newer long-term memory to score above older memory, but got {newerWeight} and {olderWeight}.");
        Assert.AreEqual(exocortex.LongTermDecayThreshold, olderWeight, 0.0001);
    }

    [TestMethod]
    public void ComputeFullMemoryWeightReturnsFiniteWeightForCurrentIdenticalMemory()
    {
        var exocortex = new TestExocortex { CustomPresentDateTime = Present };
        var memory = new CortexMemory<string>("alpha", TestExocortex.EmbeddingFor("alpha"), Present) { Type = CortexMemoryType.Core };
        exocortex.Memories.Add(memory);

        var weight = exocortex.ComputeFullMemoryWeight(memory, memory.EmbeddingVectors);

        Assert.IsFalse(double.IsNaN(weight));
        Assert.AreEqual(1d, weight, 0.0001);
    }

    [TestMethod]
    public void CortexMemoryDistanceSpaceReturnsZeroForIdenticalEmbeddings()
    {
        var exocortex = new TestExocortex { CustomPresentDateTime = Present };
        var memoryOne = new CortexMemory<string>("alpha", TestExocortex.EmbeddingFor("alpha"), Present) { Type = CortexMemoryType.Core };
        var memoryTwo = new CortexMemory<string>("also alpha", TestExocortex.EmbeddingFor("alpha"), Present) { Type = CortexMemoryType.Core };

        var distance = new CortexMemoryDistanceSpace<string>(exocortex).ComputeDistance(0, 1, memoryOne, memoryTwo);

        Assert.AreEqual(0d, distance, 0.0001);
    }

    [TestMethod]
    public async Task AddMemoryAsyncYieldsCoreAndReactionForFirstMemory()
    {
        var exocortex = new TestExocortex { CustomPresentDateTime = Present };

        var memories = await CollectAsync(exocortex.AddMemoryAsync("alpha", CancellationToken.None));

        Assert.AreEqual(2, memories.Count);
        Assert.AreEqual(CortexMemoryType.Core, memories[0].Type);
        Assert.AreEqual("alpha", memories[0].Content);
        Assert.AreEqual(CortexMemoryType.Reaction, memories[1].Type);
        Assert.AreEqual("reaction: alpha", memories[1].Content);
    }

    [TestMethod]
    public async Task AddMemoryAsyncPassesCancellationTokenToModelCalls()
    {
        using var cancellationTokenSource = new CancellationTokenSource();
        var exocortex = new TestExocortex { CustomPresentDateTime = Present };

        await CollectAsync(exocortex.AddMemoryAsync("alpha", cancellationTokenSource.Token));

        Assert.IsTrue(exocortex.EmbeddingTokens.Count >= 2);
        Assert.IsTrue(exocortex.EmbeddingTokens.All(token => token == cancellationTokenSource.Token));
        Assert.IsTrue(exocortex.ReactionTokens.Count >= 1);
        Assert.IsTrue(exocortex.ReactionTokens.All(token => token == cancellationTokenSource.Token));
    }

    [TestMethod]
    public async Task AddMemoryAsyncIncludesRelevantLongTermMemoryInReactionContext()
    {
        var exocortex = new TestExocortex
        {
            CustomPresentDateTime = Present,
            ShortTermMemoryDuration = TimeSpan.FromHours(8),
        };

        exocortex.Memories.Add(new CortexMemory<string>("alpha old", TestExocortex.EmbeddingFor("alpha"), Present.AddDays(-2)) { Type = CortexMemoryType.Core });

        await CollectAsync(exocortex.AddMemoryAsync("alpha", CancellationToken.None));

        Assert.IsTrue(exocortex.LastReactionContext.Any(memory => memory.Content == "alpha old"));
    }

    private static async Task<List<T>> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var results = new List<T>();
        await foreach (var item in source)
            results.Add(item);

        return results;
    }

    private sealed class TestExocortex : Exocortex<string>
    {
        public List<CancellationToken> EmbeddingTokens { get; } = [];

        public List<CancellationToken> ReactionTokens { get; } = [];

        public List<CortexMemory<string>> LastReactionContext { get; private set; } = [];

        public override Task<string> SummarizeMemoryInNewContext(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
        {
            return Task.FromResult($"summary: {memory.Content}");
        }

        public override Task<string> ReactToMemoryAsync(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
        {
            ReactionTokens.Add(cancellationToken);
            LastReactionContext = workingMemories.ToList();
            return Task.FromResult($"reaction: {memory.Content}");
        }

        public override Task<float[]> GenerateEmbeddingAsync(string memoryContent, CancellationToken cancellationToken = default)
        {
            EmbeddingTokens.Add(cancellationToken);
            return Task.FromResult(EmbeddingFor(memoryContent));
        }

        public static float[] EmbeddingFor(string memoryContent)
        {
            return memoryContent switch
            {
                "alpha" => [1f, 0f, 0f],
                "alpha old" => [1f, 0f, 0f],
                "newer" => [1f, 0f, 0f],
                "older" => [1f, 0f, 0f],
                _ => [0f, 1f, 0f],
            };
        }
    }
}