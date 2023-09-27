namespace OwlCore.AI.Exocortex;

/// <summary>
/// Defines the type of memory in the Exocortex.
/// </summary>
public enum CortexMemoryType
{
    /// <summary>
    /// Core memories represent the foundational experiences or knowledge.
    /// These are typically the original memories the system interacts with or receives.
    /// </summary>
    Core,

    /// <summary>
    /// Memories that have been recalled in the context of other related memories.
    /// These represent reframed or reinterpreted versions of original or previously recalled memories,
    /// taking into account additional context from clusters of related memories.
    /// </summary>
    Recollection,

    /// <summary>
    /// Reactions represent the system's direct responses or reactions to new memories or inputs.
    /// These are generated as a result of processing and understanding new experiences in the light of existing memories.
    /// </summary>
    Reaction,
}
