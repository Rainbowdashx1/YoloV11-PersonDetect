using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using OpenCvSharp;
using YoloPerson.VideoCapture;

namespace BenchmarkMethods.BenchMarksModels
{
    [MemoryDiagnoser]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [RankColumn]
    public class BenchLetterbox
    {
        private Mat testMat1080p;
        private Mat testMat720p;
        private Mat testMat4K;
        private Mat letterboxBuffer;
        private ProcessFrame processFrame;

        private const int TargetWidth = 640;
        private const int TargetHeight = 640;

        [GlobalSetup]
        public void Setup()
        {
            // Simular diferentes resoluciones de video
            testMat1080p = new Mat(new Size(1920, 1080), MatType.CV_8UC3, new Scalar(100, 150, 200));
            testMat720p = new Mat(new Size(1280, 720), MatType.CV_8UC3, new Scalar(100, 150, 200));
            testMat4K = new Mat(new Size(3840, 2160), MatType.CV_8UC3, new Scalar(100, 150, 200));
            
            // Buffer pre-alocado para todos los métodos
            letterboxBuffer = new Mat(new Size(TargetWidth, TargetHeight), MatType.CV_8UC3);
            
            processFrame = new ProcessFrame();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            testMat1080p?.Dispose();
            testMat720p?.Dispose();
            testMat4K?.Dispose();
            letterboxBuffer?.Dispose();
            processFrame?.DisposeBuffers();
        }

        // ============ Tests 1080p ============
        [Benchmark(Baseline = true)]
        [BenchmarkCategory("1080p")]
        public void Original_1080p()
        {
            processFrame.Letterbox(testMat1080p, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }

        [Benchmark]
        [BenchmarkCategory("1080p")]
        public void Optimized_1080p()
        {
            processFrame.LetterboxOptimized(testMat1080p, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }

        // ============ Tests 720p ============
        [Benchmark]
        [BenchmarkCategory("720p")]
        public void Original_720p()
        {
            processFrame.Letterbox(testMat720p, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }

        [Benchmark]
        [BenchmarkCategory("720p")]
        public void Optimized_720p()
        {
            processFrame.LetterboxOptimized(testMat720p, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }

        // ============ Tests 4K ============
        [Benchmark]
        [BenchmarkCategory("4K")]
        public void Original_4K()
        {
            processFrame.Letterbox(testMat4K, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }

        [Benchmark]
        [BenchmarkCategory("4K")]
        public void Optimized_4K()
        {
            processFrame.LetterboxOptimized(testMat4K, letterboxBuffer, TargetWidth, TargetHeight, out _, out _, out _);
        }
    }
}
