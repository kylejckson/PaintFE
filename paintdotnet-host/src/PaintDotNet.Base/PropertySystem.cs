using System.Collections;

namespace PaintDotNet.PropertySystem;

public abstract class Property
{
    protected Property(object name, object value)
    {
        Name = name;
        Value = value;
        DefaultValue = value;
    }

    public object Name { get; }
    public object Value { get; set; }
    public object DefaultValue { get; }
    public abstract string ValueType { get; }
}

public sealed class DoubleProperty : Property
{
    public DoubleProperty(object name, double defaultValue, double minValue, double maxValue)
        : base(name, defaultValue) { MinValue = minValue; MaxValue = maxValue; }
    public double MinValue { get; }
    public double MaxValue { get; }
    public override string ValueType => "double";
}

public sealed class Int32Property : Property
{
    public Int32Property(object name, int defaultValue, int minValue, int maxValue)
        : base(name, defaultValue) { MinValue = minValue; MaxValue = maxValue; }
    public int MinValue { get; }
    public int MaxValue { get; }
    public override string ValueType => "int32";
}

public sealed class BooleanProperty : Property
{
    public BooleanProperty(object name, bool defaultValue) : base(name, defaultValue) { }
    public override string ValueType => "boolean";
}

public sealed class StaticListChoiceProperty : Property
{
    public StaticListChoiceProperty(object name, object[] valueChoices, int defaultChoiceIndex)
        : base(name, valueChoices[defaultChoiceIndex]) { ValueChoices = valueChoices; }
    public object[] ValueChoices { get; }
    public override string ValueType => "choice";
}

public sealed class PropertyCollection : IEnumerable<Property>
{
    private readonly Dictionary<object, Property> properties;
    public PropertyCollection(IEnumerable<Property> properties) =>
        this.properties = properties.ToDictionary(property => property.Name);
    public Property this[object name] => properties[name];
    public Property GetProperty(object name) => properties[name];
    public IEnumerator<Property> GetEnumerator() => properties.Values.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
