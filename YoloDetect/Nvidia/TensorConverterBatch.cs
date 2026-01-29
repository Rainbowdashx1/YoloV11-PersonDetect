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
    /// Conversión de Mat a Tensor para batch de 2 imágenes.
    /// Contiene múltiples implementaciones con diferentes niveles de optimización.
    /// </summary>
    public class TensorConverterBatch
    {
        private const float InverseNormalization = 1.0f / 255.0f;
        private static readonly ArrayPool<float> FloatPool = ArrayPool<float>.Shared;
        private static Vector256<float> normFactor = Vector256.Create(InverseNormalization);
        public static DenseTensor<float> MatToTensorParallelBatch(Mat mat1, Mat mat2)
        {
            // Validación de que ambas imágenes tienen las mismas dimensiones
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            // Conversión a RGB
            if (mat1.Channels() == 3)
            {
                Cv2.CvtColor(mat1, mat1, ColorConversionCodes.BGR2RGB);
            }
            else if (mat1.Channels() == 4)
            {
                Cv2.CvtColor(mat1, mat1, ColorConversionCodes.BGRA2RGB);
            }

            if (mat2.Channels() == 3)
            {
                Cv2.CvtColor(mat2, mat2, ColorConversionCodes.BGR2RGB);
            }
            else if (mat2.Channels() == 4)
            {
                Cv2.CvtColor(mat2, mat2, ColorConversionCodes.BGRA2RGB);
            }

            int height = mat1.Rows;
            int width = mat1.Cols;
            int channels = mat1.Channels();
            int stride = width * channels;

            if (channels != 3)
            {
                throw new ArgumentException("Solo se soportan imágenes con 3 canales.");
            }

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado.");
            }

            int singleImageSize = channels * height * width;
            float[] tensorData = new float[2 * singleImageSize];

            byte[] matData1 = new byte[height * stride];
            byte[] matData2 = new byte[height * stride];
            Marshal.Copy(mat1.Data, matData1, 0, matData1.Length);
            Marshal.Copy(mat2.Data, matData2, 0, matData2.Length);

            // Procesar ambas imágenes en paralelo
            Parallel.For(0, 2, batchIdx =>
            {
                byte[] currentMatData = batchIdx == 0 ? matData1 : matData2;
                int batchOffset = batchIdx * singleImageSize;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            int tensorIndex = batchOffset + (c * height + h) * width + w;
                            int matIndex = h * stride + w * channels + c;
                            tensorData[tensorIndex] = currentMatData[matIndex] / 255.0f;
                        }
                    }
                }
            });

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width }); ;
        }

        /// <summary>
        /// Versión híbrida batch: SIMD + Unsafe + Parallel para procesar 2 imágenes.
        /// Combina las optimizaciones de MatToTensorHybrid con procesamiento batch.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorHybridBatch(Mat mat1, Mat mat2)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            // Conversión a RGB en paralelo
            Parallel.Invoke(
                () => ConvertToRgbInPlace(mat1),
                () => ConvertToRgbInPlace(mat2)
            );

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            float[] tensorData = new float[2 * singleImageSize];

            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            // Usar GCHandle para pinear el array y poder usarlo en Parallel.Invoke
            GCHandle handle = GCHandle.Alloc(tensorData, GCHandleType.Pinned);
            try
            {
                float* dstPtr = (float*)handle.AddrOfPinnedObject().ToPointer();

                // Procesar ambas imágenes en paralelo
                Parallel.Invoke(
                    () => ProcessImageHybridInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize),
                    () => ProcessImageHybridInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize)
                );
            }
            finally
            {
                handle.Free();
            }

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width });
        }

        /// <summary>
        /// Versión híbrida batch: SIMD + Unsafe + Parallel para procesar 2 imágenes.
        /// Combina las optimizaciones de MatToTensorHybrid con procesamiento batch.
        /// con tensor reutilizable - evita allocations.
        /// </summary>
        public static unsafe void MatToTensorHybridBatch(Mat mat1, Mat mat2, DenseTensor<float> tensor)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            //// Conversión a RGB en paralelo
            //Parallel.Invoke(
            //    () => ConvertToRgbInPlace(mat1),
            //    () => ConvertToRgbInPlace(mat2)
            //);

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            Memory<float> tensorMemory = tensor.Buffer;
            MemoryHandle memHandle = tensorMemory.Pin();

            try
            {
                float* dstPtr = (float*)memHandle.Pointer;

                // Procesar ambas imágenes en paralelo - BGR→RGB inline
                Parallel.Invoke(
                    () => ProcessImageBgrToRgbInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize),
                    () => ProcessImageBgrToRgbInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize)
                );
            }
            finally
            {
                memHandle.Dispose();
            }
        }

        /// <summary>
        /// Versión V2: ELIMINA Cv2.CvtColor - hace BGR→RGB directamente en SIMD.
        /// Ahorra una pasada completa sobre cada imagen.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorHybridBatchV2(Mat mat1, Mat mat2)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            // NO llamamos a ConvertToRgbInPlace - lo hacemos inline

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            float[] tensorData = new float[2 * singleImageSize];

            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            // Usar GCHandle como V1 (que demostró ser más rápido)
            GCHandle handle = GCHandle.Alloc(tensorData, GCHandleType.Pinned);
            try
            {
                float* dstPtr = (float*)handle.AddrOfPinnedObject().ToPointer();

                // Procesar ambas imágenes en paralelo (como V1)
                // PERO sin Cv2.CvtColor - BGR→RGB inline
                Parallel.Invoke(
                    () => ProcessImageBgrToRgbInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize),
                    () => ProcessImageBgrToRgbInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize)
                );
            }
            finally
            {
                handle.Free();
            }

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width });
        }

        /// <summary>
        /// Versión V3: ArrayPool + Task.Run + BGR→RGB inline.
        /// Reutiliza memoria del pool, usa Task.Run en lugar de Parallel.Invoke.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorHybridBatchV3(Mat mat1, Mat mat2)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;
            int totalSize = 2 * singleImageSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            // Pre-calcular todos los punteros ANTES de cualquier operación
            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            // Usar ArrayPool para reutilizar memoria
            float[] pooledArray = FloatPool.Rent(totalSize);
            float[] tensorData = new float[totalSize];

            GCHandle handle = GCHandle.Alloc(pooledArray, GCHandleType.Pinned);
            try
            {
                float* dstPtr = (float*)handle.AddrOfPinnedObject();

                // Task.Run en lugar de Parallel.Invoke
                var task1 = Task.Run(() => ProcessImageBgrToRgbInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize));
                var task2 = Task.Run(() => ProcessImageBgrToRgbInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize));
                Task.WaitAll(task1, task2);

                // Copiar del pool al array final
                Buffer.BlockCopy(pooledArray, 0, tensorData, 0, totalSize * sizeof(float));
            }
            finally
            {
                handle.Free();
                FloatPool.Return(pooledArray);
            }

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width });
        }

        /// <summary>
        /// Versión V4: Task.Run + BGR→RGB inline, sin ArrayPool (directo a tensorData).
        /// Compara Task.Run vs Parallel.Invoke.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorHybridBatchV4(Mat mat1, Mat mat2)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            // Pre-calcular punteros
            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            float[] tensorData = new float[2 * singleImageSize];

            // Pinear el array y obtener puntero ANTES de crear las tareas
            GCHandle handle = GCHandle.Alloc(tensorData, GCHandleType.Pinned);
            float* dstPtr = (float*)handle.AddrOfPinnedObject();

            try
            {
                // Task.Run con punteros pre-calculados
                var task1 = Task.Run(() => ProcessImageBgrToRgbInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize));
                var task2 = Task.Run(() => ProcessImageBgrToRgbInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize));

                Task.WaitAll(task1, task2);
            }
            finally
            {
                handle.Free();
            }

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width });
        }
        /// <summary>
        /// Versión batch ultra-optimizada con SIMD.
        /// </summary>
        public static unsafe DenseTensor<float> MatToTensorBatchUltraFast(Mat mat1, Mat mat2)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            // Conversión a RGB
            ConvertToRgbInPlace(mat1);
            ConvertToRgbInPlace(mat2);

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado.");
            }

            float[] tensorData = new float[2 * singleImageSize];

            // Procesar ambas imágenes en paralelo
            Parallel.Invoke(
                () => ProcessImageToTensorUnsafe(mat1, tensorData, 0, height, width, planeSize),
                () => ProcessImageToTensorUnsafe(mat2, tensorData, singleImageSize, height, width, planeSize)
            );

            return new DenseTensor<float>(tensorData, new[] { 2, channels, height, width });
        }

        /// <summary>
        /// Procesa imagen BGR directamente a tensor RGB - sin Cv2.CvtColor.
        /// Lee B,G,R y escribe R,G,B en los planos correctos.
        /// </summary>
        private static unsafe void ProcessImageBgrToRgbInternal(byte* srcPtr, int stride, float* dstPtr, int offset, int height, int width, int planeSize)
        {
            float* rPlane = dstPtr + offset;
            float* gPlane = dstPtr + offset + planeSize;
            float* bPlane = dstPtr + offset + 2 * planeSize;

            if (Avx2.IsSupported && width >= 8)
            {
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

                        // BGR en memoria: [B0,G0,R0, B1,G1,R1, ...]
                        // Leemos B (índice 0,3,6...) y lo escribimos en bPlane
                        // Leemos G (índice 1,4,7...) y lo escribimos en gPlane  
                        // Leemos R (índice 2,5,8...) y lo escribimos en rPlane

                        // Cargar B (será escrito en bPlane)
                        Vector256<int> bInt = Vector256.Create(
                            rowPtr[pixelBase], rowPtr[pixelBase + 3], rowPtr[pixelBase + 6], rowPtr[pixelBase + 9],
                            rowPtr[pixelBase + 12], rowPtr[pixelBase + 15], rowPtr[pixelBase + 18], rowPtr[pixelBase + 21]);

                        // Cargar G (será escrito en gPlane)
                        Vector256<int> gInt = Vector256.Create(
                            rowPtr[pixelBase + 1], rowPtr[pixelBase + 4], rowPtr[pixelBase + 7], rowPtr[pixelBase + 10],
                            rowPtr[pixelBase + 13], rowPtr[pixelBase + 16], rowPtr[pixelBase + 19], rowPtr[pixelBase + 22]);

                        // Cargar R (será escrito en rPlane)
                        Vector256<int> rInt = Vector256.Create(
                            rowPtr[pixelBase + 2], rowPtr[pixelBase + 5], rowPtr[pixelBase + 8], rowPtr[pixelBase + 11],
                            rowPtr[pixelBase + 14], rowPtr[pixelBase + 17], rowPtr[pixelBase + 20], rowPtr[pixelBase + 23]);

                        // Almacenar en orden RGB
                        Avx.Store(rPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(rInt), normFactor));
                        Avx.Store(gPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(gInt), normFactor));
                        Avx.Store(bPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(bInt), normFactor));
                    }

                    // Procesar píxeles restantes - BGR→RGB inline
                    for (; w < width; w++)
                    {
                        int pixelIdx = w * 3;
                        int dstIdx = rowOffset + w;
                        // BGR: [0]=B, [1]=G, [2]=R
                        bPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        rPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
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
                        // BGR: [0]=B, [1]=G, [2]=R
                        bPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        rPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                    }
                });
            }
        }

        /// <summary>
        /// Método interno que aplica las optimizaciones híbridas (SIMD + Parallel) a una imagen.
        /// </summary>
        private static unsafe void ProcessImageHybridInternal(byte* srcPtr, int stride, float* dstPtr, int offset, int height, int width, int planeSize)
        {
            float* rPlane = dstPtr + offset;
            float* gPlane = dstPtr + offset + planeSize;
            float* bPlane = dstPtr + offset + 2 * planeSize;

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ConvertToRgbInPlace(Mat mat)
        {
            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            }
            else if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            }
        }
    }
}
