using System.Text.Json;
using OwlCore.AI.Exocortex.AppModels;
using OwlCore.AI.Exocortex.DataModels;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class ApplicationModelTests
{
    private static readonly DateTime Present = new(2026, 5, 17, 12, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public void CortexMemoryDataRoundTripsWithoutPolymorphicTypes()
    {
        var memories = new List<CortexMemoryData<string>>
        {
            new()
            {
                Id = "core-1",
                Type = CortexMemoryType.Core,
                Content = "core memory",
                CreationTimestamp = Present,
                EmbeddingVectors = [1f, 0f, 0f],
            },
            new()
            {
                Id = "recollection-1",
                Type = CortexMemoryType.Recollection,
                Content = "recalled memory",
                CreationTimestamp = Present.AddMinutes(1),
                EmbeddingVectors = [0f, 1f, 0f],
                RecalledMemoryIds = ["core-1"],
            },
        };

        var json = JsonSerializer.Serialize(memories);
        var roundTripped = JsonSerializer.Deserialize<List<CortexMemoryData<string>>>(json);

        Assert.IsNotNull(roundTripped);
        Assert.AreEqual(2, roundTripped.Count);
        Assert.AreEqual("core-1", roundTripped[0].Id);
        Assert.AreEqual(CortexMemoryType.Recollection, roundTripped[1].Type);
        Assert.AreEqual("core-1", roundTripped[1].RecalledMemoryIds.Single());
    }

    [TestMethod]
    public void CortexMemoryAppModelWrapsDataModel()
    {
        var data = CreateCoreData();
        var appModel = new CortexMemoryAppModel<string>(data);

        Assert.AreSame(data, appModel.Data);
        Assert.AreEqual(data.Id, appModel.Id);
        Assert.AreEqual(data.Type, appModel.Type);
        Assert.AreEqual(data.Content, appModel.Content);
        Assert.AreEqual(data.CreationTimestamp, appModel.CreationTimestamp);
        CollectionAssert.AreEqual(data.EmbeddingVectors, appModel.EmbeddingVectors.ToArray());
    }

    [TestMethod]
    public async Task RecollectionAppModelResolvesReferencesThroughApplicationBoundary()
    {
        var coreMemory = new CortexMemoryAppModel<string>(CreateCoreData());
        var memoriesById = new Dictionary<string, ICortexMemory<string>>
        {
            [coreMemory.Id] = coreMemory,
        };

        var recollectionData = CreateRecollectionData(coreMemory.Id);
        var recollection = new RecollectionCortexMemoryAppModel<string>(
            recollectionData,
            (memoryId, _) => Task.FromResult(memoriesById.TryGetValue(memoryId, out var memory) ? memory : null));

        var recalled = new List<ICortexMemory<string>>();
        await foreach (var memory in recollection.GetRecalledMemoriesAsync())
            recalled.Add(memory);

        Assert.AreSame(recollectionData, recollection.Data);
        Assert.AreEqual(coreMemory.Id, recollection.RecalledMemoryIds.Single());
        Assert.AreEqual(1, recalled.Count);
        Assert.AreSame(coreMemory, recalled[0]);
    }

    [TestMethod]
    public void RecollectionAppModelRequiresRecollectionData()
    {
        var data = CreateCoreData();

        Assert.ThrowsException<ArgumentException>(() => new RecollectionCortexMemoryAppModel<string>(data, (_, _) => Task.FromResult<ICortexMemory<string>?>(null)));
    }

    private static CortexMemoryData<string> CreateCoreData()
    {
        return new CortexMemoryData<string>
        {
            Id = "core-1",
            Type = CortexMemoryType.Core,
            Content = "core memory",
            CreationTimestamp = Present,
            EmbeddingVectors = [1f, 0f, 0f],
        };
    }

    private static CortexMemoryData<string> CreateRecollectionData(string recalledMemoryId)
    {
        return new CortexMemoryData<string>
        {
            Id = "recollection-1",
            Type = CortexMemoryType.Recollection,
            Content = "recalled memory",
            CreationTimestamp = Present.AddMinutes(1),
            EmbeddingVectors = [0f, 1f, 0f],
            RecalledMemoryIds = [recalledMemoryId],
        };
    }
}
