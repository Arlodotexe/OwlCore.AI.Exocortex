using System;
using System.Linq;

namespace OwlCore.AI.Exocortex;

public class SemanticLogicGates
{
    /// <summary>
    /// Represents the Semantic Void operation, returning a vector that denotes the absence of any semantic information.
    /// </summary>
    /// <param name="v">
    /// The input vector. While it's provided as a parameter, its values are not directly used by the function. The length is used to produce a vector array of the same size.
    /// Example: Given an input vector representing the concept "apple", the output would be a zero vector, indicating the absence of any semantic meaning.
    /// </param>
    /// <example>
    /// Input: "apple"
    /// Expected Output: "absence of semantic meaning"
    /// 
    /// Input: "music"
    /// Expected Output: "absence of semantic meaning"
    /// 
    /// Input: "happiness"
    /// Expected Output: "absence of semantic meaning"
    /// </example>
    /// <returns>
    /// A zero vector of the same dimensionality as the input vector, representing the complete absence of semantic information.
    /// Example: For an input vector of dimensionality 3, the output would be [0, 0, 0].
    /// </returns>
    /// <remarks>
    /// The S-VOID operation is useful in scenarios where you want to explicitly represent the concept of "nothingness" or absence in a semantic space. It provides a neutral point with no direction or magnitude in the space.
    /// </remarks>
    public float[] S_VOID(params float[] v)
    {
        return v.Select(x => 0f).ToArray();
    }

    /// <summary>
    /// Computes the Semantic AND of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-AND.</param>
    /// <returns>The combined vector using Semantic AND operation.</returns>
    /// <example>
    /// Input: "car" and "engine"
    /// Expected Output: "vehicle"
    /// 
    /// Input: "apple" and "banana"
    /// Expected Output: "fruit"
    /// 
    /// Input: "flower" and "red"
    /// Expected Output: "rose"
    /// </example>
    /// <remarks>
    /// This function computes the intersection of the given vectors in a semantic space. 
    /// It operates on vectors of the same dimensionality and returns a vector where each dimension represents the minimum value across all input vectors for that dimension.
    /// Example:
    ///   - Input Vectors: [0.8, 0.2, 0.5], [0.6, 0.9, 0.3], [0.4, 0.7, 0.1]
    ///   - Output Vector: [0.4, 0.2, 0.1]
    /// </remarks>
    public static float[] S_AND(params float[][] vectors)
    {
        if (vectors.Length == 0)
            throw new ArgumentException();

        int dimension = vectors[0].Length;
        if (vectors.Any(v => v.Length != dimension))
            throw new ArgumentException("All vectors must have the same dimension.");

        float[] result = new float[dimension];

        for (int i = 0; i < dimension; i++)
        {
            result[i] = vectors.Min(v => v[i]);
        }

        return result;
    }

    /// <summary>
    /// Computes the Semantic OR of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-OR.</param>
    /// <returns>The combined vector using Semantic OR operation.</returns>
    /// <example>
    /// Input: "sunny" or "warm"
    /// Expected Output: "pleasant"
    /// 
    /// Input: "music" or "art"
    /// Expected Output: "creative"
    /// 
    /// Input: "dog" or "cat"
    /// Expected Output: "pet"
    /// </example>
    /// <remarks>
    /// This function computes the union of the given vectors in a semantic space. 
    /// It operates on vectors of the same dimensionality and returns a vector where each dimension represents the maximum value across all input vectors for that dimension.
    /// Example:
    ///   - Input Vectors: [0.8, 0.2, 0.5], [0.6, 0.9, 0.3], [0.4, 0.7, 0.1]
    ///   - Output Vector: [0.8, 0.9, 0.5]
    /// </remarks>
    public static float[] S_OR(params float[][] vectors)
    {
        if (vectors.Length == 0)
            throw new ArgumentException();

        int dimension = vectors[0].Length;
        if (vectors.Any(v => v.Length != dimension))
            throw new ArgumentException("All vectors must have the same dimension.");

        float[] result = new float[dimension];

        for (int i = 0; i < dimension; i++)
        {
            result[i] = vectors.Max(v => v[i]);
        }

        return result;
    }

    /// <summary>
    /// Computes the Semantic NOT of the provided vector.
    /// </summary>
    /// <param name="vector">The vector to negate using S-NOT.</param>
    /// <returns>The negated vector using Semantic NOT operation.</returns>
    /// <example>
    /// Input: "happy"
    /// Expected Output: "unhappy"
    /// 
    /// Input: "good"
    /// Expected Output: "bad"
    /// 
    /// Input: "open"
    /// Expected Output: "closed"
    /// </example>
    /// <remarks>
    /// This function negates a single vector in a semantic space. 
    /// It inverts each dimension of the input vector by subtracting it from 1.
    /// Example:
    ///   - Input Vector: [0.8, 0.2, 0.5]
    ///   - Output Vector: [0.2, 0.8, 0.5]
    /// </remarks>
    public static float[] S_NOT(float[] vector)
    {
        return vector.Select(v => 1 - v).ToArray();
    }

    /// <summary>
    /// Computes the Semantic NAND of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-NAND.</param>
    /// <returns>The combined vector using Semantic NAND operation.</returns>
    /// <example>
    /// Input: "bird" and "fly"
    /// Expected Output: "non-flying creature"
    /// 
    /// Input: "computer" and "waterproof"
    /// Expected Output: "non-waterproof device"
    /// 
    /// Input: "book" and "read"
    /// Expected Output: "non-readable material"
    /// </example>
    /// <remarks>
    /// This function computes the negation of the Semantic AND operation on the given vectors in a semantic space. 
    /// It effectively inverts the output of the Semantic AND.
    /// Example:
    ///   - Input Vectors: [0.8, 0.2, 0.5], [0.6, 0.9, 0.3], [0.4, 0.7, 0.1]
    ///   - Output Vector: [0.6, 0.8, 0.9]
    /// </remarks>
    public static float[] S_NAND(params float[][] vectors)
    {
        return S_NOT(S_AND(vectors));
    }

    /// <summary>
    /// Computes the Semantic NOR of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-NOR.</param>
    /// <returns>The combined vector using Semantic NOR operation.</returns>
    /// <example>
    /// Input: "cat" or "fish"
    /// Expected Output: "animal"
    /// 
    /// Input: "book" or "spoon"
    /// Expected Output: "object"
    /// 
    /// Input: "hot" or "cold"
    /// Expected Output: "temperature"
    /// </example>
    /// <remarks>
    /// This function computes the negation of the Semantic OR operation on the given vectors in a semantic space. 
    /// It effectively inverts the output of the Semantic OR.
    /// Example:
    ///   - Input Vectors: [0.8, 0.2, 0.5], [0.6, 0.9, 0.3], [0.4, 0.7, 0.1]
    ///   - Output Vector: [0.2, 0.1, 0.5]
    /// </remarks>
    public static float[] S_NOR(params float[][] vectors)
    {
        return S_NOT(S_OR(vectors));
    }

    /// <summary>
    /// Computes the Semantic XOR of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-XOR.</param>
    /// <returns>The combined vector using Semantic XOR operation.</returns>
    /// <example>
    /// Input: "hot" or "cold"
    /// Expected Output: "temperature"
    /// 
    /// Input: "apple" or "banana" or "orange"
    /// Expected Output: "fruit"
    /// 
    /// Input: "day" or "night"
    /// Expected Output: "time"
    /// </example>
    /// <remarks>
    /// This function computes the exclusive OR (XOR) of the given binary vectors in a semantic space. 
    /// It operates on vectors of the same dimensionality and returns a vector where each dimension represents the XOR result of the input vectors for that dimension.
    /// Example:
    ///   - Input Binary Vectors: [1, 0, 1], [0, 1, 0], [1, 1, 0]
    ///   - Output Vector: [0, 0, 1]
    /// </remarks>
    public static float[] S_XOR(params float[][] vectors)
    {
        if (vectors.Length == 0)
            throw new ArgumentException();

        int dimension = vectors[0].Length;
        if (vectors.Any(v => v.Length != dimension))
            throw new ArgumentException("All vectors must have the same dimension.");

        float[] result = new float[dimension];

        for (int i = 0; i < dimension; i++)
        {
            int sum = vectors.Sum(v => (int)v[i]);
            result[i] = (sum % 2 == 0) ? 0 : 1;
        }

        return result;
    }

    /// <summary>
    /// Computes the Semantic XNOR of the provided vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine using S-XNOR.</param>
    /// <returns>The combined vector using Semantic XNOR operation.</returns>
    /// <example>
    /// Input: "happy" or "sad"
    /// Expected Output: "emotional"
    /// 
    /// Input: "true" or "false"
    /// Expected Output: "logical"
    /// 
    /// Input: "right" or "left"
    /// Expected Output: "directional"
    /// </example>
    /// <remarks>
    /// This function computes the exclusive NOR (XNOR) of the given binary vectors in a semantic space. 
    /// It operates on vectors of the same dimensionality and returns a vector where each dimension represents the XNOR result of the input vectors for that dimension.
    /// Example:
    ///   - Input Binary Vectors: [1, 0, 1], [0, 1, 0], [1, 1, 0]
    ///   - Output Vector: [1, 1, 0]
    /// </remarks>
    public static float[] S_XNOR(params float[][] vectors)
    {
        return S_NOT(S_XOR(vectors));
    }

    /// <summary>
    /// Computes the Semantic LIKE operation between two vectors.
    /// </summary>
    /// <param name="vectorA">The first input vector.</param>
    /// <param name="vectorB">The second input vector.</param>
    /// <returns>The cosine similarity between the two vectors.</returns>
    /// <example>
    /// Input: "apple" and "fruit"
    /// Expected Output: A value close to 1 indicating high similarity.
    /// 
    /// Input: "apple" and "car"
    /// Expected Output: A value close to 0 indicating low similarity.
    /// </example>
    /// <remarks>
    /// This function computes the cosine similarity between two vectors, indicating how similar they are in a semantic space.
    /// </remarks>
    public static float S_LIKE(float[] vectorA, float[] vectorB)
    {
        var dotProduct = vectorA.Zip(vectorB, (a, b) => a * b).Sum();
        var magnitudeA = Math.Sqrt(vectorA.Sum(a => a * a));
        var magnitudeB = Math.Sqrt(vectorB.Sum(b => b * b));

        return (float)(dotProduct / (magnitudeA * magnitudeB));
    }

    /// <summary>
    /// Computes the Semantic NOTLIKE operation between two vectors.
    /// </summary>
    /// <param name="vectorA">The first input vector.</param>
    /// <param name="vectorB">The second input vector.</param>
    /// <returns>The cosine distance between the two vectors.</returns>
    /// <example>
    /// Input: "apple" and "fruit"
    /// Expected Output: A value close to 0 indicating low difference.
    /// 
    /// Input: "apple" and "car"
    /// Expected Output: A value closer to 1 indicating high difference.
    /// </example>
    /// <remarks>
    /// This function computes the cosine distance between two vectors, indicating how different they are in a semantic space.
    /// </remarks>
    public static float S_NOTLIKE(float[] vectorA, float[] vectorB)
    {
        return 1 - S_LIKE(vectorA, vectorB);
    }

    /// <summary>
    /// Performs the Semantic IMPLY operation on two vectors.
    /// </summary>
    /// <param name="vectorA">The first input vector representing a concept. E.g., "rain", "homework", "sun".</param>
    /// <param name="vectorB">The second input vector representing another concept that could be implied by vectorA. E.g., "umbrella", "stress", "sunglasses".</param>
    /// <returns>
    /// A vector in the semantic space that captures the implication of vectorA onto vectorB, representing the nuanced relationship between them.
    /// </returns>
    /// <example>
    /// Input: "rain" implies "umbrella"
    /// Expected Output: "needing an umbrella because of rain"
    /// 
    /// Input: "student" implies "homework"
    /// Expected Output: "doing homework as a student"
    /// 
    /// Input: "sun" implies "sunglasses"
    /// Expected Output: "wearing sunglasses on a sunny day"
    /// </example>
    /// <remarks>
    /// The Semantic IMPLY operation captures the indirect relationship between two concepts.
    /// Rather than producing a binary result, it navigates the rich semantic space to provide a vector that encapsulates the implied relationship between the inputs.
    /// This operation is crucial for understanding and making associations in a continuous semantic space.
    /// </remarks>
    public static float[] S_IMPLY(float[] vectorA, float[] vectorB)
    {
        return S_OR(S_NOT(vectorA), vectorB);
    }

    /// <summary>
    /// Rotates a vector in the semantic space towards a specified target vector, changing the cosine distance to another vector.
    /// </summary>
    /// <param name="vector">The input vector to be rotated.</param>
    /// <param name="targetVector">The target vector to which the input vector should be rotated.</param>
    /// <param name="angleRadians">The maximum angle in radians by which to rotate the vector in each step.</param>
    /// <returns>The rotated vector in the semantic space.</returns>
    /// <example>
    /// Input Vector: "happy"
    /// Target Vector: "sad"
    /// Angle (Radians): π/4 (45 degrees)
    /// Expected Output: "moving from a state of happiness towards sadness"
    /// 
    /// Input Vector: "warm"
    /// Target Vector: "cold"
    /// Angle (Radians): -π/6 (-30 degrees)
    /// Expected Output: "transitioning from warm to cold"
    /// 
    /// Input Vector: "beginning"
    /// Target Vector: "end"
    /// Angle (Radians): π/3 (60 degrees)
    /// Expected Output: "progressing from the beginning to the end"
    /// </example>
    /// <remarks>
    /// This function rotates a vector in the semantic space towards a specified target vector, allowing for shifts in meaning, perspective, or context.
    /// The angle in radians determines the maximum extent of rotation in each step:
    /// - Positive angles rotate the vector counterclockwise.
    /// - Negative angles rotate the vector clockwise.
    /// </remarks>
    public static float[] S_ROTATE(float[] vector, float[] targetVector, float angleRadians)
    {
        // GPT-generated code. Needs optimization.
        // ---
        // Ensure the input vector and target vector are normalized
        float magnitude = (float)Math.Sqrt(vector.Sum(v => v * v));
        if (Math.Abs(magnitude - 1) > float.Epsilon)
        {
            // Normalize the input vector
            vector = vector.Select(v => v / magnitude).ToArray();
        }

        float targetMagnitude = (float)Math.Sqrt(targetVector.Sum(v => v * v));
        if (Math.Abs(targetMagnitude - 1) > float.Epsilon)
        {
            // Normalize the target vector
            targetVector = targetVector.Select(v => v / targetMagnitude).ToArray();
        }

        int dimension = vector.Length;
        float[] rotatedVector = new float[dimension];

        // Calculate the angle between the input vector and the target vector
        float dotProduct = vector.Zip(targetVector, (a, b) => a * b).Sum();
        float angleBetween = (float)Math.Acos(Math.Clamp(dotProduct, -1.0, 1.0));

        // Calculate the step size based on the specified angle
        float stepSize = Math.Min(angleRadians, angleBetween);

        // Perform the rotation towards the target vector
        for (int i = 0; i < dimension; i++)
        {
            float x = vector[i];
            float y = (float)Math.Cos(stepSize) * vector[i] - (float)Math.Sin(stepSize) * targetVector[i];
            rotatedVector[i] = y;
        }

        return rotatedVector;
    }
}
