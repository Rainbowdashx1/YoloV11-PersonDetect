using Dia2Lib;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Diagnostics;
using YoloPerson.Nvidia;
using YoloPerson.PreProcess;

namespace YoloPerson.VideoCapture
{
    internal class Capture
    {
        readonly string videoPath;
        readonly string videoProcessPath;
        readonly ProcessFrame process;
        readonly SessionGpu session;
        readonly Preprocessed prePro;
        public Capture(string videoPath, string videoProcessPath, string modelPath) 
        {
            this.videoPath = videoPath;
            this.videoProcessPath = videoProcessPath;
            process = new ProcessFrame();
            session = new SessionGpu(modelPath);
            prePro = new Preprocessed();
        }
        public void run()
        {
            using var videoCapture = new OpenCvSharp.VideoCapture(videoPath);
            if (!videoCapture.IsOpened())
            {
                Console.WriteLine("No se pudo abrir el video.");
                return;
            }

            int fps = (int)videoCapture.Fps;
            int frameWidth = (int)videoCapture.FrameWidth;
            int frameHeight = (int)videoCapture.FrameHeight;

            using var videoWriter = new OpenCvSharp.VideoWriter(
                videoProcessPath,
                FourCC.XVID,
                fps,
                new OpenCvSharp.Size(frameWidth, frameHeight)
            );

            if (!videoWriter.IsOpened())
            {
                Console.WriteLine("No se pudo abrir el escritor de video.");
                return;
            }

            Mat frame = new Mat();
            int currentFrame = 0;

            while (videoCapture.Read(frame))
            {
                currentFrame++;
                if (frame.Empty())
                    break;

                List<Detection> detections = ProcessFrame(frame);

                DrawDetections(frame, detections);
                videoWriter.Write(frame);

                Cv2.ImShow("Cuadro Actual", frame);

                if (Cv2.WaitKey(1) >= 0)
                    break;
            }

            Cv2.DestroyAllWindows();
        }
        private List<Detection> ProcessFrame(Mat frame)
        {
            float r;
            int padX, padY;
            var matframeLetterbox = process.Letterbox(frame, 640, 640,out r, out padX, out padY);
            Tensor<float>? output0 = session.SessionRun(matframeLetterbox);
            return prePro.PreproccessedOutput(output0, padX, padY, r);
        }
        private void DrawDetections(Mat frame, List<Detection> detections)
        {
            foreach (var detection in detections)
            {
                int x1 = (int)detection.X1;
                int y1 = (int)detection.Y1;
                int x2 = (int)detection.X2;
                int y2 = (int)detection.Y2;

                Cv2.Rectangle(frame, new OpenCvSharp.Point(x1, y1), new OpenCvSharp.Point(x2, y2), Scalar.Red, 2);

                string label = $"Clase {detection.ClassId} ({detection.Score:P1})";
                Cv2.PutText(frame, label, new OpenCvSharp.Point(x1, y1 - 10), HersheyFonts.HersheySimplex, 0.5, Scalar.Yellow, 1);
            }
        }
    }
}
