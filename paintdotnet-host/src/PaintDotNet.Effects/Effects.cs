using System.Drawing;
using PaintDotNet.PropertySystem;

namespace PaintDotNet.Effects;

[Flags]
public enum EffectFlags : long
{
    None = 0,
    Configurable = 1
}

public static class SubmenuNames
{
    public static string Artistic => "Artistic";
    public static string Blurs => "Blurs";
    public static string Distort => "Distort";
    public static string Noise => "Noise";
    public static string Photo => "Photo";
    public static string Render => "Render";
    public static string Stylize => "Stylize";
}

public sealed class EffectEnvironmentParameters
{
    public PdnRegion Selection { get; set; } = new(new Rectangle(0, 0, 1, 1));
    public PdnRegion GetSelection(Rectangle bounds) => Selection;
}

public abstract class Effect
{
    protected Effect(string name, Image? image, string submenuName, EffectFlags flags)
    {
        Name = name;
        SubmenuName = submenuName;
        Flags = flags;
    }

    public string Name { get; }
    public string SubmenuName { get; }
    public EffectFlags Flags { get; }
    protected EffectEnvironmentParameters EnvironmentParameters { get; } = new();
}

public abstract class Effect<TToken> : Effect
{
    protected Effect(string name, Image? image, string submenuName, EffectFlags flags)
        : base(name, image, submenuName, flags) { }

    protected RenderArgs SrcArgs { get; private set; } = null!;
    protected RenderArgs DstArgs { get; private set; } = null!;
    protected TToken Token { get; private set; } = default!;
    protected bool IsCancelRequested => false;

    protected void PaintFeSetRenderInfo(TToken token, RenderArgs destination, RenderArgs source)
    {
        Token = token;
        DstArgs = destination;
        SrcArgs = source;
    }
}

public sealed class PropertyBasedEffectConfigToken
{
    public PropertyBasedEffectConfigToken(PropertyCollection properties) => Properties = properties;
    public PropertyCollection Properties { get; }
    public Property GetProperty(object name) => Properties.GetProperty(name);
    public void SetPropertyValue(object name, object value) => Properties.GetProperty(name).Value = value;
}

public abstract class PropertyBasedEffect : Effect<PropertyBasedEffectConfigToken>
{
    protected PropertyBasedEffect(string name, Image? image, string submenuName, EffectFlags flags)
        : base(name, image, submenuName, flags) { }

    protected abstract PropertyCollection OnCreatePropertyCollection();
    protected abstract void OnRender(Rectangle[] renderRects, int startIndex, int length);

    public PropertyCollection PaintFeCreateProperties() => OnCreatePropertyCollection();

    public void PaintFeRender(
        RenderArgs source,
        RenderArgs destination,
        PropertyBasedEffectConfigToken token,
        Rectangle selection)
    {
        PaintFeSetRenderInfo(token, destination, source);
        EnvironmentParameters.Selection = new PdnRegion(selection);
        var rectangles = new[] { source.Bounds };
        OnRender(rectangles, 0, rectangles.Length);
    }
}
