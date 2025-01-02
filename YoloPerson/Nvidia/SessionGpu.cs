using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace YoloPerson.Nvidia
{
    public class SessionGpu
    {
        public InferenceSession session;
        public SessionGpu(string modelPath) 
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA(0);
            session = new InferenceSession(modelPath, sessionOptions);
        }
        public Tensor<float>? SessionRun(Mat matframeLetterbox) 
        {
            DenseTensor<float> inputTensor = MatToTensor(matframeLetterbox);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
            return results.First(r => r.Name == "output0").AsTensor<float>();
        }
        private DenseTensor<float> MatToTensor(Mat letterboxMat)
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
    }
}
