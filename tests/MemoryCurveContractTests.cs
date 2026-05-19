using System;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class MemoryCurveContractTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public void ShortTermDecayThresholdDecreasesAsExocortexAgeIncreases()
    {
        var younger = CreateAgedExocortex(yearsOld: 5);
        var older = CreateAgedExocortex(yearsOld: 70);

        Assert.IsTrue(younger.ShortTermDecayThreshold > older.ShortTermDecayThreshold);
        Assert.IsTrue(older.ShortTermDecayThreshold > older.LongTermDecayThreshold);
    }

    [TestMethod]
    public void RecencyAtShortTermBoundaryEqualsDynamicThreshold()
    {
        var exocortex = CreateAgedExocortex(yearsOld: 30);
        var boundaryTimestamp = Present - exocortex.ShortTermMemoryDuration;

        var actual = exocortex.ComputeRecencyWeight(boundaryTimestamp);

        Assert.AreEqual(exocortex.ShortTermDecayThreshold, actual, 0.00001);
    }

    [TestMethod]
    public void LongTermRecencyNeverFallsBelowConfiguredThreshold()
    {
        var exocortex = CreateAgedExocortex(yearsOld: 70);
        var veryOldTimestamp = Present.AddYears(-200);

        var actual = exocortex.ComputeRecencyWeight(veryOldTimestamp);

        Assert.IsTrue(actual >= exocortex.LongTermDecayThreshold);
        Assert.IsTrue(actual <= exocortex.ShortTermDecayThreshold);
    }

    private static RecordingTextExocortex CreateAgedExocortex(int yearsOld)
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            ShortTermMemoryDuration = TimeSpan.FromMinutes(25),
            LongTermDecayThreshold = 0.1f,
        };

        exocortex.AddExisting("seed", Present.AddYears(-yearsOld));
        return exocortex;
    }
}
