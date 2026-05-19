using System;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class WorkingMemorySelectionTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public void RelevantLongTermMemoryOutranksOrthogonalLongTermMemory()
    {
        var exocortex = CreateOldExocortex();
        var similar = exocortex.AddExisting("old-similar", Present.AddYears(-60));
        var orthogonal = exocortex.AddExisting("old-orthogonal", Present.AddYears(-60));
        var queryEmbedding = exocortex.EmbeddingFor("query");

        var similarScore = exocortex.ComputeFullMemoryWeight(similar, queryEmbedding);
        var orthogonalScore = exocortex.ComputeFullMemoryWeight(orthogonal, queryEmbedding);

        Assert.IsTrue(similarScore > orthogonalScore, $"Expected similar score {similarScore} to exceed orthogonal score {orthogonalScore}.");
    }

    [TestMethod]
    public void RelevantLongTermMemoryCanOutrankRecentOrthogonalNoise()
    {
        var exocortex = CreateOldExocortex();
        var similar = exocortex.AddExisting("old-similar", Present.AddYears(-60));
        var recentNoise = exocortex.AddExisting("recent-unrelated", Present.AddMinutes(-1));
        var queryEmbedding = exocortex.EmbeddingFor("query");

        var similarScore = exocortex.ComputeFullMemoryWeight(similar, queryEmbedding);
        var recentNoiseScore = exocortex.ComputeFullMemoryWeight(recentNoise, queryEmbedding);

        Assert.IsTrue(similarScore > recentNoiseScore, $"Expected similar score {similarScore} to exceed recent noise score {recentNoiseScore}.");
    }

    [TestMethod]
    public void TypeWeightScalesScoreAfterCurveAndRelevance()
    {
        var exocortex = new RecordingTextExocortex { CustomPresentDateTime = Present };
        var memory = exocortex.AddExisting("query", Present);
        var queryEmbedding = exocortex.EmbeddingFor("query");

        Assert.AreEqual(1d, exocortex.ComputeFullMemoryWeight(memory, queryEmbedding), 0.00001);

        exocortex.CoreMemoryWeight = 0.5d;

        Assert.AreEqual(0.5d, exocortex.ComputeFullMemoryWeight(memory, queryEmbedding), 0.00001);
    }

    private static RecordingTextExocortex CreateOldExocortex()
    {
        var exocortex = new RecordingTextExocortex { CustomPresentDateTime = Present };
        exocortex.AddExisting("seed", Present.AddYears(-70));
        return exocortex;
    }
}
