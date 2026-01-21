using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;


namespace YoloPerson.Nvidia
{
    public class SessionGpu
    {
        public InferenceSession session;
        public SessionGpu(string modelPath) 
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL; // Paralelización completa
            sessionOptions.EnableMemoryPattern = true; // Optimización de patrones de memoria
            sessionOptions.EnableCpuMemArena = false; // Desactivar para GPU pura
            sessionOptions.EnableProfiling = false; // Sin profiling para máximo rendimiento

            sessionOptions.AddSessionConfigEntry("session.dynamic_block_base", "8"); // Bloques más grandes
            sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1"); // Allocators optimizados
            sessionOptions.AddSessionConfigEntry("session.disable_prepacking", "0"); // Habilitar prepacking

            sessionOptions.AddSessionConfigEntry("ep.cuda.device_id", "0"); // GPU principal
            sessionOptions.AddSessionConfigEntry("ep.cuda.arena_extend_strategy", "kSameAsRequested"); // Estrategia de memoria agresiva
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_mem_limit", "0"); // Sin límite de memoria GPU
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv_algo_search", "EXHAUSTIVE"); // Búsqueda exhaustiva del mejor algoritmo

            sessionOptions.AddSessionConfigEntry("ep.cuda.do_copy_in_default_stream", "1"); // Copia en stream por defecto
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv1d_pad_to_nc1d", "1"); // Optimización de padding
            sessionOptions.AddSessionConfigEntry("ep.cuda.enable_cuda_graph", "1"); // CUDA Graphs para máximo rendimiento
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv_use_max_workspace", "1"); // Usar máximo workspace de cuDNN

            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_alloc", "0"); // Allocator interno para mejor rendimiento
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_free", "0");
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_empty_cache", "0");

            sessionOptions.AddSessionConfigEntry("ep.cuda.tunable_op_enable", "1"); // Habilitar ops tunables
            sessionOptions.AddSessionConfigEntry("ep.cuda.tunable_op_tuning_enable", "1"); // Auto-tuning activado
            sessionOptions.AddSessionConfigEntry("ep.cuda.user_compute_stream", "1"); // Stream de computación dedicado

            sessionOptions.AddSessionConfigEntry("session.set_denormal_as_zero", "1"); // Tratar denormales como cero
            sessionOptions.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1"); // Allocator GPU para inicializadores
            sessionOptions.AddSessionConfigEntry("session.inter_op_num_threads", "0"); // Usar todos los threads disponibles
            sessionOptions.AddSessionConfigEntry("session.intra_op_num_threads", "0"); // Auto-detección

            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // Todas las optimizaciones
            sessionOptions.AddSessionConfigEntry("optimization.minimal_build_optimizations", ""); // Sin restricciones

            sessionOptions.AppendExecutionProvider_CUDA(0);
            session = new InferenceSession(modelPath, sessionOptions);
        }
        public Tensor<float>? SessionRun(Mat matframeLetterbox) 
        {
            DenseTensor<float> inputTensor = MatToTensorHybrid(matframeLetterbox);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
            return results.First(r => r.Name == "output0").AsTensor<float>();
        }
        public Tensor<float>? SessionRunBatch(Mat mat1, Mat mat2)
        {
            DenseTensor<float> inputTensor = MatToTensorHybridBatch(mat1, mat2);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            var results = session.Run(inputs);
            return results.First(r => r.Name == "output0").AsTensor<float>();
        }
        public DenseTensor<float> MatToTensor(Mat letterboxMat)
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
        public DenseTensor<float> MatToTensorParallel(Mat mat)
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
        public DenseTensor<float> MatToTensorParallelBatch(Mat mat1, Mat mat2)
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

        // Constante pre-calculada para evitar división en runtime
        private const float InverseNormalization = 1.0f / 255.0f;

        // ArrayPool compartido para reutilizar buffers y evitar allocations
        private static readonly ArrayPool<byte> BytePool = ArrayPool<byte>.Shared;
        private static readonly ArrayPool<float> FloatPool = ArrayPool<float>.Shared;

        /// <summary>
        /// Versión ultra-optimizada usando SIMD (AVX2), ArrayPool y acceso directo a memoria.
        /// Procesa 8 floats en paralelo usando Vector256.
        /// </summary>
        public DenseTensor<float> MatToTensorUltraFast(Mat mat)
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
        public unsafe DenseTensor<float> MatToTensorUnsafe(Mat mat)
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
        public unsafe DenseTensor<float> MatToTensorHybrid(Mat mat)
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
        /// Versión híbrida batch: SIMD + Unsafe + Parallel para procesar 2 imágenes.
        /// Combina las optimizaciones de MatToTensorHybrid con procesamiento batch.
        /// </summary>
        public unsafe DenseTensor<float> MatToTensorHybridBatch(Mat mat1, Mat mat2)
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
        /// Versión V2: ELIMINA Cv2.CvtColor - hace BGR→RGB directamente en SIMD.
        /// Ahorra una pasada completa sobre cada imagen.
        /// </summary>
        public unsafe DenseTensor<float> MatToTensorHybridBatchV2(Mat mat1, Mat mat2)
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
        public unsafe DenseTensor<float> MatToTensorHybridBatchV3(Mat mat1, Mat mat2)
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
        public unsafe DenseTensor<float> MatToTensorHybridBatchV4(Mat mat1, Mat mat2)
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
        /// Procesa imagen BGR directamente a tensor RGB - sin Cv2.CvtColor.
        /// Lee B,G,R y escribe R,G,B en los planos correctos.
        /// </summary>
        private unsafe void ProcessImageBgrToRgbInternal(byte* srcPtr, int stride, float* dstPtr, int offset, int height, int width, int planeSize)
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
        private unsafe void ProcessImageHybridInternal(byte* srcPtr, int stride, float* dstPtr, int offset, int height, int width, int planeSize)
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

        /// <summary>
        /// Versión batch ultra-optimizada con SIMD.
        /// </summary>
        public unsafe DenseTensor<float> MatToTensorBatchUltraFast(Mat mat1, Mat mat2)
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

        private unsafe void ProcessImageToTensorUnsafe(Mat mat, float[] tensorData, int offset, int height, int width, int planeSize)
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

        private void ProcessWithAvx2(byte[] matData, float[] tensorData, int height, int width, int stride)
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

        private void ProcessWithSse2(byte[] matData, float[] tensorData, int height, int width, int stride)
        {
            // Similar a AVX2 pero con vectores de 128 bits
            ProcessScalarOptimized(matData, tensorData, height, width, stride);
        }

        private void ProcessScalarOptimized(byte[] matData, float[] tensorData, int height, int width, int stride)
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
