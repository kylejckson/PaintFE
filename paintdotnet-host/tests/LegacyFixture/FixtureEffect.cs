using System.Drawing;
using PaintDotNet;
using PaintDotNet.Effects;
using PaintDotNet.PropertySystem;

namespace LegacyFixture;

public sealed class FixtureEffect()
    : PropertyBasedEffect("Legacy Fixture", null, SubmenuNames.Stylize, EffectFlags.Configurable)
{
    protected override PropertyCollection OnCreatePropertyCollection() =>
        new([
            new DoubleProperty("Amount", 0.5, 0, 1),
            new Int32Property("Iterations", 1, 1, 10),
            new BooleanProperty("Enabled", true),
            new StaticListChoiceProperty("Mode", ["Normal", "Invert"], 0)
        ]);

    protected override void OnRender(Rectangle[] renderRects, int startIndex, int length)
    {
        var amount = (float)(double)Token.GetProperty("Amount").Value;
        for (var i = startIndex; i < startIndex + length; i++)
        {
            var rect = renderRects[i];
            for (var y = rect.Top; y < rect.Bottom; y++)
            for (var x = rect.Left; x < rect.Right; x++)
            {
                var source = SrcArgs.Surface[x, y];
                DstArgs.Surface[x, y] = ColorBgra.FromBgra(
                    (byte)(source.B * amount),
                    (byte)(source.G * amount),
                    (byte)(source.R * amount),
                    source.A);
            }
        }
    }
}
