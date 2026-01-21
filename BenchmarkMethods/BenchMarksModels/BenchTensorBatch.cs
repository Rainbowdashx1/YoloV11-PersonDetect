using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using YoloPerson.Nvidia;

namespace BenchmarkMethods.BenchMarksModels
{
    [MemoryDiagnoser]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [RankColumn]
    public class BenchTensorBatch
    {
        private Mat testMat640_1 = null!;
        private Mat testMat640_2 = null!;
        private SessionGpu sessionGpu = null!;
        private string ModelPath = null!;

        [GlobalSetup]
        public void Setup()
        {
            ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");

            // Crear Mat de 640x640 con variación para simular imágenes reales
            testMat640_1 = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            testMat640_2 = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(50, 100, 150));

            Cv2.Randn(testMat640_1, new Scalar(128, 128, 128), new Scalar(50, 50, 50));
            Cv2.Randn(testMat640_2, new Scalar(128, 128, 128), new Scalar(50, 50, 50));

            sessionGpu = new SessionGpu(ModelPath);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            testMat640_1?.Dispose();
            testMat640_2?.Dispose();
            sessionGpu?.session?.Dispose();
        }

        [Benchmark(Baseline = true)]
        public DenseTensor<float> ParallelBatch_Baseline()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorParallelBatch(clone1, clone2);
        }

        [Benchmark]
        public DenseTensor<float> BatchUltraFast()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorBatchUltraFast(clone1, clone2);
        }

        [Benchmark]
        public DenseTensor<float> HybridBatch_V1()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorHybridBatch(clone1, clone2);
        }

        [Benchmark]
        public DenseTensor<float> HybridBatch_V2_NoCvtColor()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorHybridBatchV2(clone1, clone2);
        }

        [Benchmark]
        public DenseTensor<float> HybridBatch_V3_ArrayPool()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorHybridBatchV3(clone1, clone2);
        }

        [Benchmark]
        public DenseTensor<float> HybridBatch_V4_TaskRun()
        {
            using var clone1 = testMat640_1.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorHybridBatchV4(clone1, clone2);
        }
    }
}
