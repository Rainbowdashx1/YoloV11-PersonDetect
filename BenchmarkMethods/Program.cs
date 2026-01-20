using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using BenchmarkMethods.BenchMarksModels;

public class Program
{
    private static void Main(string[] args)
    {
        var config = DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator);//Required for ONNX nuget, not in release mode
        var summaryLetter = BenchmarkRunner.Run<BenchLetterbox>(config);
        var summaryResize = BenchmarkRunner.Run<BenchResize>(config);
    }
}