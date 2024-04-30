namespace System;

/// <inheritdoc/>
internal static partial class MathEx
{
    public static T Clamp<T>(T value, T min, T max) where T : IComparable<T>
    {
        if (value.CompareTo(min) < 0) return min;
        else if (value.CompareTo(max) > 0) return max;
        else return value;
    }
}