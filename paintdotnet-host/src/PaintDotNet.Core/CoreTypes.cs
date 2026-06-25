using System.Drawing;
using System.Runtime.InteropServices;

namespace PaintDotNet
{

public interface IPluginSupportInfo
{
    string Author { get; }
    string Copyright { get; }
    string DisplayName { get; }
    Version Version { get; }
    Uri WebsiteUri { get; }
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct ColorBgra
{
    public byte B;
    public byte G;
    public byte R;
    public byte A;

    public static ColorBgra FromBgra(byte b, byte g, byte r, byte a) =>
        new() { B = b, G = g, R = r, A = a };
}

public sealed class Surface : IDisposable
{
    private readonly ColorBgra[] pixels;

    public Surface(int width, int height)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        Width = width;
        Height = height;
        pixels = new ColorBgra[checked(width * height)];
    }

    public int Width { get; }
    public int Height { get; }
    public Rectangle Bounds => new(0, 0, Width, Height);

    public ColorBgra this[int x, int y]
    {
        get => pixels[checked(y * Width + x)];
        set => pixels[checked(y * Width + x)] = value;
    }

    public void CopySurface(Surface source, Point destination, Rectangle sourceRect)
    {
        for (var y = 0; y < sourceRect.Height; y++)
        for (var x = 0; x < sourceRect.Width; x++)
            this[destination.X + x, destination.Y + y] = source[sourceRect.X + x, sourceRect.Y + y];
    }

    public void Dispose() { }
}

public sealed class RenderArgs : IDisposable
{
    public RenderArgs(Surface surface) => Surface = surface;
    public Surface Surface { get; }
    public Rectangle Bounds => Surface.Bounds;
    public void Dispose() { }
}

public sealed class PdnRegion
{
    public PdnRegion(Rectangle bounds) => Bounds = bounds;
    public Rectangle Bounds { get; }
}

public sealed class EffectEnvironmentParameters
{
    public PdnRegion Selection { get; set; } = new(new Rectangle(0, 0, 1, 1));
    public PdnRegion GetSelection(Rectangle bounds) => Selection;
}
}
