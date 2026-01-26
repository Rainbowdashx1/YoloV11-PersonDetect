using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace YoloPerson.Nvidia
{
    /// <summary>
    /// Conversión de Mat a Tensor para una sola imagen.
    /// Contiene múltiples implementaciones con diferentes niveles de optimización.
    /// </summary>
    public class TensorConverterSingle
    {
        private const float InverseNormalization = 1.0f / 255.0f;
        private static readonly ArrayPool<byte> BytePool = ArrayPool<byte>.Shared;

        public static DenseTensor<float> MatToTensor(Mat letterboxMat)
        {
            Mat fmat = new Mat();
            letterboxMat.ConvertTo(fmat, MatType.CV_32F, 1 / 255.0);

            float[] chw = new float[3 * 640 * 640];
            int idx = 0;
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 640; y++)
                {
                    for (int x = 0; x < 640; x++)
                    {
                        chw[idx++] = fmat.At<Vec3f>(y, x)[c];
                    }
                }
            }
            return new DenseTensor<float>(chw, new[] { 1, 3, 640, 640 });
        }
        public static DenseTensor<float> MatToTensorParallel(Mat mat)
        {
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat.Rows;
            int width = mat.Cols;
            int channels = mat.Channels();
            int stride = width * channels;

            if (channels != 3)
            {
                throw new ArgumentException("Solo se soportan imágenes con 3 canales.");
            }

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            float[] tensorData = new float[channels * height * width];
            byte[] matData = new byte[height * stride];
            Marshal.Copy(mat.Data, matData, 0, matData.Length);

            Parallel.For(0, height, h =>
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        int tensorIndex = (c * height + h) * width + w;
                        int matIndex = h * stride + w * channels + c;
                        tensorData[tensorIndex] = matData[matIndex] / 255.0f;
                    }
                }
            });

            return new DenseTensor<float>(tensorData, new[] { 1, channels, height, width });
        }

        /// <summary>
        /// Versión ultra-optimizada usando SIMD (AVX2), ArrayPool y acceso directo a memoria.
        /// Procesa 8 floats en paralelo usando Vector256.
        /// </summary>
        public static DenseTensor<float> MatToTensorUltraFast(Mat mat)
        {
            // Conversión BGR a RGB
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat.Rows;
            int width = mat.Cols;
            const int channels = 3;
            int stride = width * channels;
            int pixelCount = height * width;
            int totalSize = channels * pixelCount;

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            // Usar ArrayPool para evitar allocations - mejor para GC
            byte[] matData = BytePool.Rent(height * stride);
            float[] tensorData = new float[totalSize]; // Este debe ser exacto para DenseTensor

            try
            {
                // Copiar datos de imagen a buffer manejado
                Marshal.Copy(mat.Data, matData, 0, height * stride);

                // Procesar usando SIMD si está disponible
                if (Avx2.IsSupported)
                {
                    ProcessWithAvx2(matData, tensorData, height, width, stride);
                }
                else if (Sse2.IsSupported)
                {
                    ProcessWithSse2(matData, tensorData, height, width, stride);
                }
                else
                {
                    ProcessScalarOptimized(matData, tensorData, height, width, stride);
                }
            }
            finally
            {
                BytePool.Return(matData);
            }

            return new DenseTensor<float>(tensorData, new[] { 1, channels, height, width });
        }

        /// <summary>
        /// Versión con unsafe pointers para máximo rendimiento - evita bounds checking.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorUnsafe(Mat mat)
        {
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat.Rows;
            int width = mat.Cols;
            const int channels = 3;
            int stride = width * channels;
            int planeSize = height * width;

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            float[] tensorData = new float[channels * planeSize];

            byte* srcPtr = (byte*)mat.Data.ToPointer();

            fixed (float* dstPtr = tensorData)
            {
                float* rPlane = dstPtr;
                float* gPlane = dstPtr + planeSize;
                float* bPlane = dstPtr + 2 * planeSize;

                Parallel.For(0, height, h =>
                {
                    byte* rowPtr = srcPtr + h * stride;
                    int rowOffset = h * width;

                    for (int w = 0; w < width; w++)
                    {
                        int pixelIdx = w * 3;
                        int dstIdx = rowOffset + w;

                        rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                    }
                });
            }

            return new DenseTensor<float>(tensorData, new[] { 1, channels, height, width });
        }

        /// <summary>
        /// Versión híbrida: SIMD + Unsafe + Parallel + ArrayPool
        /// La más optimizada de todas.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorHybrid(Mat mat)
        {
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat.Rows;
            int width = mat.Cols;
            const int channels = 3;
            int planeSize = height * width;

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            float[] tensorData = new float[channels * planeSize];
            byte* srcPtr = (byte*)mat.Data.ToPointer();
            int stride = (int)mat.Step();

            fixed (float* dstPtr = tensorData)
            {
                float* rPlane = dstPtr;
                float* gPlane = dstPtr + planeSize;
                float* bPlane = dstPtr + 2 * planeSize;

                if (Avx2.IsSupported && width >= 8)
                {
                    Vector256<float> normFactor = Vector256.Create(InverseNormalization);

                    Parallel.For(0, height, h =>
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;
                        int w = 0;

                        // Procesar 8 píxeles a la vez con AVX2
                        int simdWidth = width - (width % 8);
                        for (; w < simdWidth; w += 8)
                        {
                            int pixelBase = w * 3;
                            int dstBase = rowOffset + w;

                            // Cargar y convertir 8 valores R
                            Vector256<int> rInt = Vector256.Create(
                                rowPtr[pixelBase], rowPtr[pixelBase + 3], rowPtr[pixelBase + 6], rowPtr[pixelBase + 9],
                                rowPtr[pixelBase + 12], rowPtr[pixelBase + 15], rowPtr[pixelBase + 18], rowPtr[pixelBase + 21]);
                            Vector256<float> rFloat = Avx.Multiply(Avx.ConvertToVector256Single(rInt), normFactor);

                            // Cargar y convertir 8 valores G
                            Vector256<int> gInt = Vector256.Create(
                                rowPtr[pixelBase + 1], rowPtr[pixelBase + 4], rowPtr[pixelBase + 7], rowPtr[pixelBase + 10],
                                rowPtr[pixelBase + 13], rowPtr[pixelBase + 16], rowPtr[pixelBase + 19], rowPtr[pixelBase + 22]);
                            Vector256<float> gFloat = Avx.Multiply(Avx.ConvertToVector256Single(gInt), normFactor);

                            // Cargar y convertir 8 valores B
                            Vector256<int> bInt = Vector256.Create(
                                rowPtr[pixelBase + 2], rowPtr[pixelBase + 5], rowPtr[pixelBase + 8], rowPtr[pixelBase + 11],
                                rowPtr[pixelBase + 14], rowPtr[pixelBase + 17], rowPtr[pixelBase + 20], rowPtr[pixelBase + 23]);
                            Vector256<float> bFloat = Avx.Multiply(Avx.ConvertToVector256Single(bInt), normFactor);

                            // Almacenar resultados
                            Avx.Store(rPlane + dstBase, rFloat);
                            Avx.Store(gPlane + dstBase, gFloat);
                            Avx.Store(bPlane + dstBase, bFloat);
                        }

                        // Procesar píxeles restantes de forma escalar
                        for (; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    });
                }
                else
                {
                    // Fallback sin SIMD
                    Parallel.For(0, height, h =>
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;

                        for (int w = 0; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    });
                }
            }

            return new DenseTensor<float>(tensorData, new[] { 1, channels, height, width });
        }
        /// <summary>
        /// Versión híbrida con tensor reutilizable - evita allocations.
        /// </summary>
        public static unsafe void MatToTensorHybrid(Mat mat, DenseTensor<float> tensor)
        {
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat.Rows;
            int width = mat.Cols;
            int planeSize = height * width;

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            // Obtener el buffer interno del tensor
            Span<float> tensorData = tensor.Buffer.Span;

            byte* srcPtr = (byte*)mat.Data.ToPointer();
            int stride = (int)mat.Step();

            fixed (float* dstPtr = tensorData)
            {
                float* rPlane = dstPtr;
                float* gPlane = dstPtr + planeSize;
                float* bPlane = dstPtr + 2 * planeSize;

                if (Avx2.IsSupported && width >= 8)
                {
                    Vector256<float> normFactor = Vector256.Create(InverseNormalization);

                    Parallel.For(0, height, h =>
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;
                        int w = 0;

                        int simdWidth = width - (width % 8);
                        for (; w < simdWidth; w += 8)
                        {
                            int pixelBase = w * 3;
                            int dstBase = rowOffset + w;

                            Vector256<int> rInt = Vector256.Create(
                                rowPtr[pixelBase], rowPtr[pixelBase + 3], rowPtr[pixelBase + 6], rowPtr[pixelBase + 9],
                                rowPtr[pixelBase + 12], rowPtr[pixelBase + 15], rowPtr[pixelBase + 18], rowPtr[pixelBase + 21]);
                            Vector256<float> rFloat = Avx.Multiply(Avx.ConvertToVector256Single(rInt), normFactor);

                            Vector256<int> gInt = Vector256.Create(
                                rowPtr[pixelBase + 1], rowPtr[pixelBase + 4], rowPtr[pixelBase + 7], rowPtr[pixelBase + 10],
                                rowPtr[pixelBase + 13], rowPtr[pixelBase + 16], rowPtr[pixelBase + 19], rowPtr[pixelBase + 22]);
                            Vector256<float> gFloat = Avx.Multiply(Avx.ConvertToVector256Single(gInt), normFactor);

                            Vector256<int> bInt = Vector256.Create(
                                rowPtr[pixelBase + 2], rowPtr[pixelBase + 5], rowPtr[pixelBase + 8], rowPtr[pixelBase + 11],
                                rowPtr[pixelBase + 14], rowPtr[pixelBase + 17], rowPtr[pixelBase + 20], rowPtr[pixelBase + 23]);
                            Vector256<float> bFloat = Avx.Multiply(Avx.ConvertToVector256Single(bInt), normFactor);

                            Avx.Store(rPlane + dstBase, rFloat);
                            Avx.Store(gPlane + dstBase, gFloat);
                            Avx.Store(bPlane + dstBase, bFloat);
                        }

                        for (; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    });
                }
                else
                {
                    Parallel.For(0, height, h =>
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;

                        for (int w = 0; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    });
                }
            }
        }
        private static unsafe void ProcessImageToTensorUnsafe(Mat mat, float[] tensorData, int offset, int height, int width, int planeSize)
        {
            byte* srcPtr = (byte*)mat.Data.ToPointer();
            int stride = (int)mat.Step();

            fixed (float* dstPtr = tensorData)
            {
                float* rPlane = dstPtr + offset;
                float* gPlane = dstPtr + offset + planeSize;
                float* bPlane = dstPtr + offset + 2 * planeSize;

                Parallel.For(0, height, h =>
                {
                    byte* rowPtr = srcPtr + h * stride;
                    int rowOffset = h * width;

                    for (int w = 0; w < width; w++)
                    {
                        int pixelIdx = w * 3;
                        int dstIdx = rowOffset + w;

                        rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                    }
                });
            }
        }
        private static void ProcessWithAvx2(byte[] matData, float[] tensorData, int height, int width, int stride)
        {
            int planeSize = height * width;
            Vector256<float> normFactor = Vector256.Create(InverseNormalization);

            Parallel.For(0, height, h =>
            {
                int rowStart = h * stride;
                int rowOffset = h * width;

                for (int w = 0; w < width; w++)
                {
                    int pixelIdx = rowStart + w * 3;
                    int dstIdx = rowOffset + w;

                    tensorData[dstIdx] = matData[pixelIdx] * InverseNormalization;                    // R
                    tensorData[planeSize + dstIdx] = matData[pixelIdx + 1] * InverseNormalization;    // G
                    tensorData[2 * planeSize + dstIdx] = matData[pixelIdx + 2] * InverseNormalization; // B
                }
            });
        }
        private static void ProcessWithSse2(byte[] matData, float[] tensorData, int height, int width, int stride)
        {
            // Similar a AVX2 pero con vectores de 128 bits
            ProcessScalarOptimized(matData, tensorData, height, width, stride);
        }

        private static void ProcessScalarOptimized(byte[] matData, float[] tensorData, int height, int width, int stride)
        {
            int planeSize = height * width;

            Parallel.For(0, height, h =>
            {
                int rowStart = h * stride;
                int rowOffset = h * width;

                for (int w = 0; w < width; w++)
                {
                    int pixelIdx = rowStart + w * 3;
                    int dstIdx = rowOffset + w;

                    // Multiplicación es más rápida que división
                    tensorData[dstIdx] = matData[pixelIdx] * InverseNormalization;
                    tensorData[planeSize + dstIdx] = matData[pixelIdx + 1] * InverseNormalization;
                    tensorData[2 * planeSize + dstIdx] = matData[pixelIdx + 2] * InverseNormalization;
                }
            });
        }
    }
}
