using System.Drawing;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.Loader;
using System.Text.Json;
using PaintDotNet;
using PaintDotNet.Effects;
using PaintDotNet.PropertySystem;

const int ProtocolVersion = 1;

if (args is ["--self-test", ..])
{
    var surface = new Surface(2, 2);
    surface[1, 1] = ColorBgra.FromBgra(1, 2, 3, 4);
    if (surface[1, 1].R != 3) throw new InvalidOperationException("Surface self-test failed");
    if (args.Length > 1)
    {
        ValidatePlatformCompatibility(args[1]);
        var loader = new PluginLoadContext(args[1]);
        var assembly = loader.LoadFromAssemblyPath(Path.GetFullPath(args[1]));
        var effectType = assembly.GetTypes().Single(type =>
            !type.IsAbstract && typeof(PropertyBasedEffect).IsAssignableFrom(type));
        var effect = (PropertyBasedEffect)Activator.CreateInstance(effectType)!;
        if (!effect.PaintFeCreateProperties().Any())
            throw new InvalidOperationException("Fixture properties were not discovered");
    }
    Console.WriteLine("PaintFE Paint.NET host self-test passed");
    return;
}

try
{
    var request = await Framing.ReadRequestAsync(Console.OpenStandardInput());
    if (request.ProtocolVersion != ProtocolVersion)
        throw new NotSupportedException($"Protocol version {request.ProtocolVersion} is not supported");
    ValidatePlatformCompatibility(request.PluginPath);
    var loader = new PluginLoadContext(request.PluginPath);
    var assembly = loader.LoadFromAssemblyPath(Path.GetFullPath(request.PluginPath));
    var effects = assembly.GetTypes()
        .Where(type => !type.IsAbstract && typeof(PropertyBasedEffect).IsAssignableFrom(type))
        .ToArray();
    if (effects.Length == 0) throw new NotSupportedException("No PropertyBasedEffect was found");

    var effectType = request.EffectType is null
        ? effects[0]
        : effects.Single(type => type.FullName == request.EffectType);
    var effect = (PropertyBasedEffect?)Activator.CreateInstance(effectType)
        ?? throw new InvalidOperationException("Effect constructor returned null");

    if (request.Command == "describe")
    {
        var describeProperties = effect.PaintFeCreateProperties();
        var response = new HostResponse(
            ProtocolVersion, true, null, effect.Name, effect.SubmenuName, effectType.FullName,
            describeProperties.Select(PropertyDescription.FromProperty).ToArray(), 0);
        await Framing.WriteResponseAsync(Console.OpenStandardOutput(), response, []);
        return;
    }

    if (request.Command != "render") throw new NotSupportedException($"Unknown command: {request.Command}");
    var expected = checked(request.Width * request.Height * 4);
    if (request.Pixels.Length != expected) throw new InvalidDataException("RGBA payload length mismatch");

    using var source = new Surface(request.Width, request.Height);
    using var destination = new Surface(request.Width, request.Height);
    var offset = 0;
    for (var y = 0; y < request.Height; y++)
    for (var x = 0; x < request.Width; x++)
    {
        var pixel = ColorBgra.FromBgra(
            request.Pixels[offset + 2], request.Pixels[offset + 1],
            request.Pixels[offset], request.Pixels[offset + 3]);
        source[x, y] = pixel;
        destination[x, y] = pixel;
        offset += 4;
    }

    var properties = effect.PaintFeCreateProperties();
    foreach (var pair in request.Parameters ?? new Dictionary<string, JsonElement>())
        SetProperty(properties.GetProperty(pair.Key), pair.Value);

    using var srcArgs = new RenderArgs(source);
    using var dstArgs = new RenderArgs(destination);
    var selection = SelectionBounds(request.Mask, request.Width, request.Height);
    effect.PaintFeRender(
        srcArgs, dstArgs, new PropertyBasedEffectConfigToken(properties),
        selection);

    var output = new byte[expected];
    offset = 0;
    for (var y = 0; y < request.Height; y++)
    for (var x = 0; x < request.Width; x++)
    {
        var pixel = destination[x, y];
        output[offset] = pixel.R;
        output[offset + 1] = pixel.G;
        output[offset + 2] = pixel.B;
        output[offset + 3] = pixel.A;
        offset += 4;
    }

    var ok = new HostResponse(ProtocolVersion, true, null, effect.Name, effect.SubmenuName,
        effectType.FullName, [], output.Length);
    await Framing.WriteResponseAsync(Console.OpenStandardOutput(), ok, output);
}
catch (Exception exception)
{
    var error = new HostResponse(ProtocolVersion, false,
        FriendlyError(exception), null, null, null, [], 0);
    await Framing.WriteResponseAsync(Console.OpenStandardOutput(), error, []);
    Environment.ExitCode = 1;
}

static void ValidatePlatformCompatibility(string pluginPath)
{
    if (OperatingSystem.IsWindows()) return;

    using var stream = File.OpenRead(pluginPath);
    using var peReader = new PEReader(stream);
    if (!peReader.HasMetadata) return;

    var metadata = peReader.GetMetadataReader();
    var references = metadata.AssemblyReferences
        .Select(handle => metadata.GetString(metadata.GetAssemblyReference(handle).Name))
        .ToHashSet(StringComparer.OrdinalIgnoreCase);
    var windowsDependency = references.FirstOrDefault(name => name is
        "System.Windows.Forms" or
        "PresentationCore" or
        "PresentationFramework" or
        "WindowsBase");

    if (windowsDependency is not null)
    {
        throw new PlatformNotSupportedException(
            $"This plugin is Windows-only (it depends on {windowsDependency}) and cannot run on " +
            $"{System.Runtime.InteropServices.RuntimeInformation.OSDescription}. " +
            "You can keep it installed, but it must be run from PaintFE on Windows.");
    }
}

static string FriendlyError(Exception exception)
{
    if (exception is ReflectionTypeLoadException typeLoad)
    {
        var details = typeLoad.LoaderExceptions
            .Where(error => error is not null)
            .Select(error => error!.Message)
            .Distinct()
            .Take(3);
        return $"Unsupported plugin API: {string.Join(" ", details)}";
    }
    return $"{exception.GetType().Name}: {exception.Message}";
}

static void SetProperty(Property property, JsonElement value)
{
    property.Value = property switch
    {
        DoubleProperty => value.GetDouble(),
        Int32Property => value.GetInt32(),
        BooleanProperty => value.GetBoolean(),
        StaticListChoiceProperty choice => choice.ValueChoices[value.GetInt32()],
        _ => throw new NotSupportedException($"Unsupported property type: {property.GetType().FullName}")
    };
}

static Rectangle SelectionBounds(byte[] mask, int width, int height)
{
    if (mask.Length != width * height) return new Rectangle(0, 0, width, height);
    var minX = width;
    var minY = height;
    var maxX = -1;
    var maxY = -1;
    for (var y = 0; y < height; y++)
    for (var x = 0; x < width; x++)
    {
        if (mask[y * width + x] == 0) continue;
        minX = Math.Min(minX, x);
        minY = Math.Min(minY, y);
        maxX = Math.Max(maxX, x);
        maxY = Math.Max(maxY, y);
    }
    return maxX < minX
        ? Rectangle.Empty
        : Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
}

sealed class PluginLoadContext(string pluginPath) : AssemblyLoadContext(isCollectible: true)
{
    private readonly string directory = Path.GetDirectoryName(Path.GetFullPath(pluginPath))!;
    protected override Assembly? Load(AssemblyName name)
    {
        if (name.Name?.StartsWith("PaintDotNet.", StringComparison.OrdinalIgnoreCase) == true)
        {
            var loaded = Default.Assemblies.FirstOrDefault(a => a.GetName().Name == name.Name);
            if (loaded is not null) return loaded;
            var shim = Path.Combine(AppContext.BaseDirectory, $"{name.Name}.dll");
            return File.Exists(shim) ? Default.LoadFromAssemblyPath(shim) : null;
        }
        var dependency = Path.Combine(directory, $"{name.Name}.dll");
        return File.Exists(dependency) ? LoadFromAssemblyPath(dependency) : null;
    }
}

sealed record HostRequest(
    int ProtocolVersion, string Command, string PluginPath, string? EffectType,
    int Width, int Height, Dictionary<string, JsonElement>? Parameters, byte[] Pixels, byte[] Mask);

sealed record HostResponse(
    int ProtocolVersion, bool Ok, string? Error, string? Name, string? Category,
    string? EffectType, PropertyDescription[] Properties, int PixelLength);

sealed record PropertyDescription(
    string Name, string Kind, JsonElement Default, double? Min, double? Max, string[] Choices)
{
    public static PropertyDescription FromProperty(Property property)
    {
        double? min = null;
        double? max = null;
        var choices = Array.Empty<string>();
        switch (property)
        {
            case DoubleProperty value:
                min = value.MinValue;
                max = value.MaxValue;
                break;
            case Int32Property value:
                min = value.MinValue;
                max = value.MaxValue;
                break;
            case StaticListChoiceProperty value:
                choices = value.ValueChoices.Select(v => v.ToString() ?? "").ToArray();
                break;
        }
        return new(property.Name.ToString() ?? string.Empty, property.ValueType,
            JsonSerializer.SerializeToElement(property.DefaultValue), min, max, choices);
    }
}

static class Framing
{
    public static async Task<HostRequest> ReadRequestAsync(Stream input)
    {
        var headerLength = await ReadInt32Async(input);
        if (headerLength is <= 0 or > 1_048_576) throw new InvalidDataException("Invalid header length");
        var header = new byte[headerLength];
        await input.ReadExactlyAsync(header);
        using var document = JsonDocument.Parse(header);
        var root = document.RootElement;
        var pixelLength = root.TryGetProperty("pixelLength", out var length) ? length.GetInt32() : 0;
        var maskLength = root.TryGetProperty("maskLength", out var maskLengthValue) ? maskLengthValue.GetInt32() : 0;
        if (pixelLength < 0) throw new InvalidDataException("Invalid pixel length");
        if (maskLength < 0) throw new InvalidDataException("Invalid mask length");
        var pixels = new byte[pixelLength];
        await input.ReadExactlyAsync(pixels);
        var mask = new byte[maskLength];
        await input.ReadExactlyAsync(mask);
        return new(
            root.GetProperty("protocolVersion").GetInt32(), root.GetProperty("command").GetString()!,
            root.GetProperty("pluginPath").GetString()!,
            root.TryGetProperty("effectType", out var type) && type.ValueKind != JsonValueKind.Null ? type.GetString() : null,
            root.TryGetProperty("width", out var width) ? width.GetInt32() : 0,
            root.TryGetProperty("height", out var height) ? height.GetInt32() : 0,
            root.TryGetProperty("parameters", out var parameters)
                ? JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(parameters) : null,
            pixels, mask);
    }

    public static async Task WriteResponseAsync(Stream output, HostResponse response, byte[] pixels)
    {
        var header = JsonSerializer.SerializeToUtf8Bytes(response,
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        await output.WriteAsync(BitConverter.GetBytes(header.Length));
        await output.WriteAsync(header);
        await output.WriteAsync(pixels);
        await output.FlushAsync();
    }

    private static async Task<int> ReadInt32Async(Stream input)
    {
        var bytes = new byte[4];
        await input.ReadExactlyAsync(bytes);
        return BitConverter.ToInt32(bytes);
    }
}
