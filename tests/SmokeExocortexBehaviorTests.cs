using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace OwlCore.AI.Exocortex.Tests;

[TestClass]
public class SmokeExocortexBehaviorTests
{
    [TestMethod]
    [TestCategory("Smoke")]
    public async Task SmallModelSmokeRecallsPriorMemory()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("OWLCORE_EXOCORTEX_SMOKE"), "1", StringComparison.OrdinalIgnoreCase))
            Assert.Inconclusive("Set OWLCORE_EXOCORTEX_SMOKE=1 to run small-model Ollama smoke tests.");

        using var httpClient = new HttpClient
        {
            BaseAddress = NormalizeBaseUri(Environment.GetEnvironmentVariable("OWLCORE_EXOCORTEX_SMOKE_OLLAMA_BASE_URL") ?? "http://Thoth-Scribe-Zero:11434"),
            Timeout = TimeSpan.FromMinutes(10),
        };
        var textModel = Environment.GetEnvironmentVariable("OWLCORE_EXOCORTEX_SMOKE_TEXT_MODEL") ?? "qwen3.5:9b";
        var embeddingModel = Environment.GetEnvironmentVariable("OWLCORE_EXOCORTEX_SMOKE_EMBEDDING_MODEL") ?? "embeddinggemma:300m";
        await AssertModelIsInstalledAsync(httpClient, textModel, CancellationToken.None);
        await AssertModelIsInstalledAsync(httpClient, embeddingModel, CancellationToken.None);

        var exocortex = new OllamaSmokeExocortex(httpClient, textModel, embeddingModel)
        {
            NumberOfDimensions = 100,
            ShortTermMemoryDuration = TimeSpan.FromHours(8),
        };
        var phrase = $"cycle-three-smoke-{Guid.NewGuid():N}";

        await CollectAsync(exocortex.AddMemoryAsync($"Remember this cycle 3 smoke phrase: {phrase}. Reply briefly that you saved it."));
        var recallMemories = await CollectAsync(exocortex.AddMemoryAsync("What is the cycle 3 smoke phrase? Answer with only the phrase."));

        Assert.IsTrue(recallMemories.Any(memory => memory.Type == CortexMemoryType.Reaction && memory.Content.Contains(phrase, StringComparison.OrdinalIgnoreCase)), "Expected the small-model smoke reaction to contain the seeded phrase.");
    }

    private static Uri NormalizeBaseUri(string url)
    {
        var normalized = url.EndsWith("/", StringComparison.Ordinal) ? url : $"{url}/";
        return new Uri(normalized, UriKind.Absolute);
    }

    private static async Task AssertModelIsInstalledAsync(HttpClient httpClient, string model, CancellationToken cancellationToken)
    {
        var tags = await httpClient.GetFromJsonAsync<TagsResponse>("api/tags", cancellationToken).ConfigureAwait(false);
        if (tags?.Models?.Any(installed => string.Equals(installed.Name, model, StringComparison.Ordinal)) != true)
            Assert.Inconclusive($"Ollama model '{model}' is not installed on the smoke host.");
    }

    private static async Task<List<T>> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var items = new List<T>();
        await foreach (var item in source)
            items.Add(item);

        return items;
    }

    private sealed class OllamaSmokeExocortex : Exocortex<string>
    {
        private readonly HttpClient _httpClient;
        private readonly string _textModel;
        private readonly string _embeddingModel;

        public OllamaSmokeExocortex(HttpClient httpClient, string textModel, string embeddingModel)
        {
            _httpClient = httpClient;
            _textModel = textModel;
            _embeddingModel = embeddingModel;
        }

        public override Task<string> SummarizeMemoryInNewContext(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
        {
            return GenerateTextAsync(BuildPrompt("Summarize the new memory using only the supplied working memories.", memory, workingMemories), cancellationToken);
        }

        public override Task<string> ReactToMemoryAsync(CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories, CancellationToken cancellationToken = default)
        {
            return GenerateTextAsync(BuildPrompt("Answer the new memory using the supplied working memories. If a requested phrase is present, return it exactly.", memory, workingMemories), cancellationToken);
        }

        public override async Task<float[]> GenerateEmbeddingAsync(string memoryContent, CancellationToken cancellationToken = default)
        {
            var response = await _httpClient.PostAsJsonAsync("api/embed", new EmbedRequest(_embeddingModel, memoryContent), cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadFromJsonAsync<EmbedResponse>(cancellationToken: cancellationToken).ConfigureAwait(false);
            var embedding = result?.Embeddings?.FirstOrDefault();
            if (embedding is null)
                throw new InvalidOperationException("Ollama smoke embedding response did not include an embedding.");

            return embedding;
        }

        private async Task<string> GenerateTextAsync(string prompt, CancellationToken cancellationToken)
        {
            var response = await _httpClient.PostAsJsonAsync("api/generate", new GenerateRequest(_textModel, prompt, Stream: false), cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadFromJsonAsync<GenerateResponse>(cancellationToken: cancellationToken).ConfigureAwait(false);
            return result?.Response ?? string.Empty;
        }

        private static string BuildPrompt(string instruction, CortexMemory<string> memory, IEnumerable<CortexMemory<string>> workingMemories)
        {
            var builder = new StringBuilder();
            builder.AppendLine(instruction);
            builder.AppendLine();
            builder.AppendLine("Working memories:");
            foreach (var workingMemory in workingMemories.OrderBy(item => item.CreationTimestamp))
                builder.AppendLine($"- [{workingMemory.Type}] {workingMemory.Content}");
            builder.AppendLine();
            builder.AppendLine("New memory:");
            builder.AppendLine(memory.Content);
            return builder.ToString();
        }
    }

    private sealed record TagsResponse([property: JsonPropertyName("models")] ModelInfo[]? Models);

    private sealed record ModelInfo([property: JsonPropertyName("name")] string Name);

    private sealed record GenerateRequest(
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("prompt")] string Prompt,
        [property: JsonPropertyName("stream")] bool Stream);

    private sealed record GenerateResponse([property: JsonPropertyName("response")] string? Response);

    private sealed record EmbedRequest(
        [property: JsonPropertyName("model")] string Model,
        [property: JsonPropertyName("input")] string Input);

    private sealed record EmbedResponse([property: JsonPropertyName("embeddings")] float[][]? Embeddings);
}
