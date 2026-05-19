using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class RollingContextContractTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public async Task FutureMemoriesAreExcludedFromReactionContext()
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            NumberOfDimensions = 100,
        };
        exocortex.AddExisting("old-similar", Present.AddHours(-1));
        exocortex.AddExisting("similar", Present.AddHours(1));

        await CollectAsync(exocortex.AddMemoryAsync("query"));

        Assert.IsTrue(exocortex.LastReactionContext.Any(memory => memory.Content == "old-similar"));
        Assert.IsFalse(exocortex.LastReactionContext.Any(memory => memory.Content == "similar"));
    }

    [TestMethod]
    public async Task SmallOldEnoughContextCanIncludeUnrelatedMemoriesButWeightsStillOrderSimilarFirst()
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            NumberOfDimensions = 100,
        };
        exocortex.AddExisting("seed", Present.AddYears(-70));
        var similar = exocortex.AddExisting("old-similar", Present.AddYears(-60));
        var orthogonal = exocortex.AddExisting("old-orthogonal", Present.AddYears(-60));
        exocortex.AddExisting("recent-unrelated", Present.AddMinutes(-1));

        await CollectAsync(exocortex.AddMemoryAsync("query"));

        Assert.IsTrue(exocortex.LastReactionContext.Any(memory => memory.Content == "old-orthogonal"));

        var queryEmbedding = exocortex.EmbeddingFor("query");
        var similarScore = exocortex.ComputeFullMemoryWeight(similar, queryEmbedding);
        var orthogonalScore = exocortex.ComputeFullMemoryWeight(orthogonal, queryEmbedding);

        Assert.IsTrue(similarScore > orthogonalScore, $"Expected similar score {similarScore} to exceed orthogonal score {orthogonalScore}.");
    }

    private static async Task<List<T>> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var items = new List<T>();
        await foreach (var item in source)
            items.Add(item);

        return items;
    }
}
