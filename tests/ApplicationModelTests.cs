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
    public async Task RecollectionAppModelResolvesReferencesThroughApplicationBoundary()
    {
        var coreMemory = new CortexMemoryAppModel<string>("core-1", CortexMemoryType.Core, "core memory", [1f, 0f, 0f], Present);
        var memoriesById = new Dictionary<string, ICortexMemory<string>>
        {
            [coreMemory.Id] = coreMemory,
        };

        var recollection = new RecollectionCortexMemoryAppModel<string>(
            "recollection-1",
            "recalled memory",
            [0f, 1f, 0f],
            Present.AddMinutes(1),
            [coreMemory.Id],
            (memoryId, _) => Task.FromResult(memoriesById.TryGetValue(memoryId, out var memory) ? memory : null));

        var recalled = new List<ICortexMemory<string>>();
        await foreach (var memory in recollection.GetRecalledMemoriesAsync())
            recalled.Add(memory);

        Assert.AreEqual(1, recalled.Count);
        Assert.AreSame(coreMemory, recalled[0]);
    }

    [TestMethod]
    public async Task RecollectionAppModelMapsToDataWithReferencedMemoryIds()
    {
        var coreMemory = new CortexMemoryAppModel<string>("core-1", CortexMemoryType.Core, "core memory", [1f, 0f, 0f], Present);
        var memoriesById = new Dictionary<string, ICortexMemory<string>>
        {
            [coreMemory.Id] = coreMemory,
        };
        var recollection = new RecollectionCortexMemoryAppModel<string>(
            "recollection-1",
            "recalled memory",
            [0f, 1f, 0f],
            Present.AddMinutes(1),
            [coreMemory.Id],
            (memoryId, _) => Task.FromResult(memoriesById.TryGetValue(memoryId, out var memory) ? memory : null));

        var data = await CortexMemoryDataMapper.ToDataAsync(recollection);

        Assert.AreEqual("recollection-1", data.Id);
        Assert.AreEqual(CortexMemoryType.Recollection, data.Type);
        Assert.AreEqual("core-1", data.RecalledMemoryIds.Single());
    }

    [TestMethod]
    public void DataMapperHydratesConcreteRecollectionForExistingEngine()
    {
        var coreData = new CortexMemoryData<string>
        {
            Id = "core-1",
            Type = CortexMemoryType.Core,
            Content = "core memory",
            CreationTimestamp = Present,
            EmbeddingVectors = [1f, 0f, 0f],
        };
        var coreMemory = CortexMemoryDataMapper.ToCortexMemory(coreData);
        var memoriesById = new Dictionary<string, CortexMemory<string>>
        {
            [coreData.Id] = coreMemory,
        };
        var recollectionData = new CortexMemoryData<string>
        {
            Id = "recollection-1",
            Type = CortexMemoryType.Recollection,
            Content = "recalled memory",
            CreationTimestamp = Present.AddMinutes(1),
            EmbeddingVectors = [0f, 1f, 0f],
            RecalledMemoryIds = [coreData.Id],
        };

        var memory = CortexMemoryDataMapper.ToCortexMemory(recollectionData, memoriesById);

        var recollection = memory as RecollectionCortexMemory<string>;
        Assert.IsNotNull(recollection);
        Assert.AreSame(coreMemory, recollection.RecalledMemories.Single());
    }
}
