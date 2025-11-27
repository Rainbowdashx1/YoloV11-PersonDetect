using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Diagnostics;
using System.Runtime.InteropServices;


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
            DenseTensor<float> inputTensor = MatToTensorParallel(matframeLetterbox);

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
            DenseTensor<float> inputTensor = MatToTensorParallelBatch(mat1, mat2);

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
    }
}
