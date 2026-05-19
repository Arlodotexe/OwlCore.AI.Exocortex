using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class LoopDiagnosticHarnessTests
{
    private static readonly DateTime Present = new(2026, 5, 18, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public async Task DiagnosticQuestionReceivesPriorLoopCoreObservations()
    {
        var exocortex = CreateLoopHarnessExocortex();
        var loopDuration = TimeSpan.FromSeconds(Math.Pow(2, 14));
        var firstObservation = "loop-observation-1 bell before courier";
        var secondObservation = "loop-observation-2 bell before courier";
        exocortex.Embeddings[firstObservation] = [1f, 0f, 0f];
        exocortex.Embeddings[secondObservation] = [1f, 0f, 0f];
        exocortex.Embeddings["diagnostic-query"] = [1f, 0f, 0f];

        exocortex.AddExisting(firstObservation, Present - TimeSpan.FromTicks(loopDuration.Ticks * 2));
        exocortex.AddExisting(secondObservation, Present - loopDuration);
        exocortex.AddExisting("reset-resident market rumor", Present - loopDuration, CortexMemoryType.Core);

        await CollectAsync(exocortex.AddMemoryAsync("diagnostic-query"));

        var workingMemories = exocortex.LastReactionContext.OfType<WorkingCortexMemory<string>>().ToList();
        Assert.IsTrue(workingMemories.Any(memory => memory.Content == firstObservation));
        Assert.IsTrue(workingMemories.Any(memory => memory.Content == secondObservation));
        Assert.IsTrue(workingMemories.Where(memory => memory.Content.StartsWith("loop-observation", StringComparison.Ordinal)).All(memory => memory.WeighedMemory.Type == CortexMemoryType.Core));
        Assert.IsFalse(workingMemories.Any(memory => memory.WeighedMemory.Type == CortexMemoryType.Recollection));
    }

    [TestMethod]
    public async Task DiagnosticQuestionDoesNotInventMissingCoreFactsIntoContext()
    {
        var exocortex = CreateLoopHarnessExocortex();
        exocortex.Embeddings["diagnostic-query"] = [1f, 0f, 0f];
        exocortex.AddExisting("loop-observation gate closes after bell", Present.AddHours(-6));

        await CollectAsync(exocortex.AddMemoryAsync("diagnostic-query"));

        Assert.IsFalse(exocortex.LastReactionContext.Any(memory => memory.Content.Contains("parents", StringComparison.OrdinalIgnoreCase)));
        Assert.IsFalse(exocortex.LastReactionContext.Any(memory => memory.Content.Contains("married", StringComparison.OrdinalIgnoreCase)));
    }

    private static RecordingTextExocortex CreateLoopHarnessExocortex()
    {
        var exocortex = new RecordingTextExocortex
        {
            CustomPresentDateTime = Present,
            NumberOfDimensions = 100,
            ShortTermMemoryDuration = TimeSpan.FromHours(1),
        };

        exocortex.AddExisting("century-age-anchor", Present.AddYears(-100));
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
