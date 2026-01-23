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
            DenseTensor<float> inputTensor = TensorConverterSingle.MatToTensorHybrid(matframeLetterbox);

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
            DenseTensor<float> inputTensor = TensorConverterBatch.MatToTensorHybridBatch(mat1, mat2);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            var results = session.Run(inputs);
            return results.First(r => r.Name == "output0").AsTensor<float>();
        }
    }
}
