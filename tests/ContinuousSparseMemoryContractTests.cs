using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class ContinuousSparseMemoryContractTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public async Task HeartbeatFilledGapChangesRollingContextComparedWithSparseGap()
    {
        var sparse = CreateGapExocortex();
        var heartbeat = CreateGapExocortex();
        heartbeat.AddExisting("heartbeat", Present.AddHours(-1));

        await CollectAsync(sparse.AddMemoryAsync("query"));
        await CollectAsync(heartbeat.AddMemoryAsync("query"));

        Assert.IsFalse(sparse.LastReactionContext.Exists(memory => memory.Content == "heartbeat"));
        Assert.IsTrue(heartbeat.LastReactionContext.Exists(memory => memory.Content == "heartbeat"));
        Assert.IsTrue(heartbeat.LastReactionContext.Count > sparse.LastReactionContext.Count);
    }

    private static RecordingTextExocortex CreateGapExocortex()
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            NumberOfDimensions = 100,
        };
        exocortex.AddExisting("seed", Present.AddDays(-2));
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
