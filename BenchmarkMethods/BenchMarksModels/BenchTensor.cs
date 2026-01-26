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
        private Mat testMat640 = null!;
        private SessionGpu sessionGpu = null!;
        private string ModelPath = null!;
        private DenseTensor<float> _reusableTensor = new(new[] { 1, 3, 640, 640 });

        [GlobalSetup]
        public void Setup()
        {
            ModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");

            // Crear Mat de 640x640 con variación para simular imagen real
            testMat640 = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            Cv2.Randn(testMat640, new Scalar(128, 128, 128), new Scalar(50, 50, 50));

            sessionGpu = new SessionGpu(ModelPath);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            testMat640?.Dispose();
            sessionGpu?.session?.Dispose();
        }

        #region Single Image Benchmarks

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensor_Original()
        {
            // Clonar para evitar modificar el original (BGR2RGB modifica in-place)
            using var clone = testMat640.Clone();
            return TensorConverterSingle.MatToTensor(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorParallel_Current()
        {
            using var clone = testMat640.Clone();
            return TensorConverterSingle.MatToTensorParallel(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorUltraFast_SIMD()
        {
            using var clone = testMat640.Clone();
            return TensorConverterSingle.MatToTensorUltraFast(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorUnsafe_Pointers()
        {
            using var clone = testMat640.Clone();
            return TensorConverterSingle.MatToTensorUnsafe(clone);
        }

        [Benchmark]
        [BenchmarkCategory("Single")]
        public DenseTensor<float> MatToTensorHybrid_Best()
        {
            using var clone = testMat640.Clone();
            return TensorConverterSingle.MatToTensorHybrid(clone);
        }


        #endregion

        #region Without Clone (Real-world scenario where Mat is consumed)



        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Parallel_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = TensorConverterSingle.MatToTensorParallel(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Hybrid_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = TensorConverterSingle.MatToTensorHybrid(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> UltraFast_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = TensorConverterSingle.MatToTensorUltraFast(mat);
            mat.Dispose();
            return result;
        }

        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Unsafe_NoClone()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            var result = TensorConverterSingle.MatToTensorUnsafe(mat);
            mat.Dispose();
            return result;
        }
        [Benchmark]
        [BenchmarkCategory("NoClone")]
        public DenseTensor<float> Hybrid_NoCloneNoTensor()
        {
            var mat = new Mat(new Size(640, 640), MatType.CV_8UC3, new Scalar(100, 150, 200));
            TensorConverterSingle.MatToTensorHybrid(mat, _reusableTensor);
            mat.Dispose();
            return _reusableTensor;
        }
        #endregion
    }
}
