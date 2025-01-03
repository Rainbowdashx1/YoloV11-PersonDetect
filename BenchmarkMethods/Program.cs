using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using BenchmarkMethods.BenchMarksModels;

public class Program
{
    private static void Main(string[] args)
    {
        var config = DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator);//Required for ONNX nuget, not in release mode
        var summary = BenchmarkRunner.Run<BenchResize>(config);
    }
}