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
    public class BenchTensor
    {
        private Mat testMat640;
        private Mat testMat640_2;
        private SessionGpu sessionGpu = null!;
        private string ModelPath;

        [GlobalSetup]
        public void Setup()
        {
            ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");
            // Crear Mat de 640x640
            testMat640 = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            testMat640_2 = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(50, 100, 150));

            // Añadir algo de variación para simular imagen real
            Cv2.Randn(testMat640, new Scalar(128, 128, 128), new Scalar(50, 50, 50));
            Cv2.Randn(testMat640_2, new Scalar(128, 128, 128), new Scalar(50, 50, 50));

            // Inicializar SessionGpu - necesario para acceder a los métodos de conversión
            sessionGpu = new SessionGpu(ModelPath);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            testMat640?.Dispose();
            testMat640_2?.Dispose();
            sessionGpu?.session?.Dispose();
        }

        #region Single Image Benchmarks

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensor_Original()
        {
            // Clonar para evitar modificar el original (BGR2RGB modifica in-place)
            using var clone = testMat640.Clone();
            return sessionGpu.MatToTensor(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorParallel_Current()
        {
            using var clone = testMat640.Clone();
            return sessionGpu.MatToTensorParallel(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorUltraFast_SIMD()
        {
            using var clone = testMat640.Clone();
            return sessionGpu.MatToTensorUltraFast(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorUnsafe_Pointers()
        {
            using var clone = testMat640.Clone();
            return sessionGpu.MatToTensorUnsafe(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorHybrid_Best()
        {
            using var clone = testMat640.Clone();
            return sessionGpu.MatToTensorHybrid(clone);
        }

        #endregion

        #region Batch Benchmarks

        //[Benchmark(Baseline = true)]
        [Benchmark]
        [BenchmarkCategory("Batch")]
        public DenseTensor<float> MatToTensorParallelBatch_Current()
        {
            using var clone1 = testMat640.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorParallelBatch(clone1, clone2);
        }

        [Benchmark]
        [BenchmarkCategory("Batch")]
        public DenseTensor<float> MatToTensorBatchUltraFast_Optimized()
        {
            using var clone1 = testMat640.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorBatchUltraFast(clone1, clone2);
        }

        [Benchmark]
        [BenchmarkCategory("Batch")]
        public DenseTensor<float> MatToTensorHybridBatch_Best()
        {
            using var clone1 = testMat640.Clone();
            using var clone2 = testMat640_2.Clone();
            return sessionGpu.MatToTensorHybridBatch(clone1, clone2);
        }

        #endregion

        #region Without Clone (Real-world scenario where Mat is consumed)

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Parallel_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = sessionGpu.MatToTensorParallel(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Hybrid_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = sessionGpu.MatToTensorHybrid(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> UltraFast_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = sessionGpu.MatToTensorUltraFast(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Unsafe_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = sessionGpu.MatToTensorUnsafe(mat);
            mat.Dispose();
            return result;
        }

        #endregion
    }
}
