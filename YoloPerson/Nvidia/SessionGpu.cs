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
            sessionOptions.AddSessionConfigEntry("session.dynamic_block_base", "4");
            sessionOptions.EnableMemoryPattern = false;
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
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

            var tensor = new DenseTensor<float>(tensorData, new[] { 1, channels, height, width });

            return tensor;
        }
    }
}
