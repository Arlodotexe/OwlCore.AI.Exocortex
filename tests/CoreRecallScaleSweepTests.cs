using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class CoreRecallScaleSweepTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    private static readonly int[] ScaleExponents = [0, 6, 12, 16, 20, 24, 28, 32];

    [TestMethod]
    public async Task SimilarCoreMemoriesRemainAvailableAcrossPowerOfTwoTimeScales()
    {
        foreach (var exponent in ScaleExponents)
        {
            var exocortex = CreateAgedExocortex(100);
            var loopMemoryContent = $"loop-clue-2^{exponent}";
            exocortex.Embeddings[loopMemoryContent] = [1f, 0f, 0f];
            var loopMemory = exocortex.AddExisting(loopMemoryContent, Present - TimeSpan.FromSeconds(Math.Pow(2, exponent)));

            await CollectAsync(exocortex.AddMemoryAsync("query"));

            Assert.IsTrue(
                exocortex.LastReactionContext.Any(memory => memory.Content == loopMemoryContent),
                $"Expected core memory at 2^{exponent} seconds to be present in reaction context.");
            Assert.IsTrue(
                exocortex.ComputeFullMemoryWeight(loopMemory, exocortex.EmbeddingFor("query")) >= exocortex.WorkingReactionMemoryWeightThreshold,
                $"Expected core memory at 2^{exponent} seconds to meet the reaction threshold.");
        }
    }

    [TestMethod]
    public void SimilarCoreMemoryOutranksOrthogonalMemoryAcrossAgeBands()
    {
        foreach (var ageYears in new[] { 0, 1, 30, 100 })
        {
            var exocortex = CreateAgedExocortex(ageYears);
            var similarContent = $"similar-age-{ageYears}";
            var orthogonalContent = $"orthogonal-age-{ageYears}";
            exocortex.Embeddings[similarContent] = [1f, 0f, 0f];
            exocortex.Embeddings[orthogonalContent] = [0f, 1f, 0f];
            var similar = exocortex.AddExisting(similarContent, Present.AddDays(-30));
            var orthogonal = exocortex.AddExisting(orthogonalContent, Present.AddDays(-30));

            var queryEmbedding = exocortex.EmbeddingFor("query");
            var similarScore = exocortex.ComputeFullMemoryWeight(similar, queryEmbedding);
            var orthogonalScore = exocortex.ComputeFullMemoryWeight(orthogonal, queryEmbedding);

            Assert.IsTrue(
                similarScore > orthogonalScore,
                $"Expected similar memory to outrank orthogonal memory for age {ageYears}; similar={similarScore}, orthogonal={orthogonalScore}.");
        }
    }

    private static RecordingTextExocortex CreateAgedExocortex(int ageYears)
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            NumberOfDimensions = 100,
        };

        if (ageYears > 0)
            exocortex.AddExisting($"age-anchor-{ageYears}", Present.AddYears(-ageYears));

        return exocortex;
    }

    private static async Task<List<T>> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var items = new List<T>();
        await foreach (var item in source)
            items.Add(item);

        return items;
    }
}
