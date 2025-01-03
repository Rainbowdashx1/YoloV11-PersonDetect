using BenchmarkDotNet.Attributes;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloPerson.Nvidia;

namespace BenchmarkMethods.BenchMarksModels
{
    [MemoryDiagnoser]

    public class BenchResize
    {
        private Mat testMat;
        private readonly SessionGpu Session = new SessionGpu("E:\\yoloperson\\YoloV11-PersonDetect\\YoloPerson\\ModelOnnx\\yolo11m.onnx");// Change it according to your project's location

        [GlobalSetup]
        public void Setup()
        {
            testMat = new Mat(new OpenCvSharp.Size(640, 640), MatType.CV_8UC3, new Scalar(0, 0, 255));
        }

        [Benchmark]
        public DenseTensor<float> OriginalMatToTensor()
        {
            return Session.MatToTensor(testMat);
        }

        [Benchmark]
        public DenseTensor<float> OptimizedMatToTensorv2()
        {
            return Session.MatToTensorParallel(testMat);
        }
    }
}
