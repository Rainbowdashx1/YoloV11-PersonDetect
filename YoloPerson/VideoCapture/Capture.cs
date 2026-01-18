using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
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
        readonly FrameRender frameRender;
        public Capture(string videoPath, string videoProcessPath, string modelPath) 
        {
            this.videoPath = videoPath;
            this.videoProcessPath = videoProcessPath;
            process = new ProcessFrame();
            session = new SessionGpu(modelPath);
            prePro = new Preprocessed();
            frameRender = new FrameRender();
        }
        public void runWithModel1Batch()
        {
            var videoCapture = VideoCapture(videoPath);
            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;

                while (videoCapture.Item1.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                        break;

                    List<Detection> detections = ProcessFrame(frame);
                    frameRender.DrawDetections(frame, detections);
                    videoCapture.Item2.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);

                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }

                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoCapture.Item1?.Dispose();
                videoCapture.Item2?.Dispose();
            }
        }
        public void runWithModel2Batch()
        {
            var videoCapture = VideoCapture(videoPath);
            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;

                while (videoCapture.Item1.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                        break;

                    List<Detection> detections = ProcessFrameBatchOverLap(frame);
                    frameRender.DrawDetections(frame, detections);
                    videoCapture.Item2.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);
                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }
                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoCapture.Item1?.Dispose();
                videoCapture.Item2?.Dispose();
            }
        }
        private (OpenCvSharp.VideoCapture, OpenCvSharp.VideoWriter) VideoCapture(string videoPath)
        {
            var videoCapture = new OpenCvSharp.VideoCapture(videoPath);
            if (!videoCapture.IsOpened())
            {
                Console.WriteLine("No se pudo abrir el video.");
                throw new Exception("No se pudo abrir el video.");
            }

            int fps = (int)videoCapture.Fps;
            int frameWidth = (int)videoCapture.FrameWidth;
            int frameHeight = (int)videoCapture.FrameHeight;

            var videoWriter = new OpenCvSharp.VideoWriter(
                videoProcessPath,
                FourCC.XVID,
                fps,
                new OpenCvSharp.Size(frameWidth, frameHeight)
            );

            if (!videoWriter.IsOpened())
            {
                Console.WriteLine("No se pudo abrir el escritor de video.");
                throw new Exception("No se pudo abrir el escritor de video.");
            }

            return (videoCapture, videoWriter);
        }
        private List<Detection> ProcessFrame(Mat frame)
        {
            float r;
            int padX, padY;
            var matframeLetterbox = process.Letterbox(frame, 640, 640,out r, out padX, out padY);
            Tensor<float>? output0 = session.SessionRun(matframeLetterbox);
            return prePro.PreproccessedOutput(output0, padX, padY, r);
        }
        private List<Detection> ProcessFrameBatchOverLap(Mat frame)
        {
            int overlapPixels = 150; // Solapamiento configurable
            int halfWidth = frame.Width / 2;

            int leftWidth = halfWidth + overlapPixels;
            int rightStart = halfWidth - overlapPixels;
            int rightWidth = frame.Width - rightStart;

            using Mat leftRegion = new Mat(frame, new Rect(0, 0, leftWidth, frame.Height));
            using Mat rightRegion = new Mat(frame, new Rect(rightStart, 0, rightWidth, frame.Height));

            float r1, r2;
            int padX1, padY1, padX2, padY2;
    
            var leftLetterbox = process.Letterbox(leftRegion, 640, 640, out r1, out padX1, out padY1);
            var rightLetterbox = process.Letterbox(rightRegion, 640, 640, out r2, out padX2, out padY2);

            Tensor<float>? outputSession = session.SessionRunBatch(leftLetterbox, rightLetterbox);
            var (leftDetections, rightDetections) = prePro.PreproccessedOutputBatchOptimized(
                outputSession,
                padX1, padY1, r1,
                padX2, padY2, r2
            );
  
            for (int i = 0; i < rightDetections.Count; i++)
            {
                var det = rightDetections[i];
                rightDetections[i] = new Detection(
                    det.X1 + rightStart,
                    det.Y1,
                    det.X2 + rightStart,
                    det.Y2,
                    det.Score,
                    det.ClassId
                );
            }
            // Combinar y eliminar duplicados en la zona solapada
            var allDetections = MergeOverlappingDetections(
                leftDetections,
                rightDetections,
                halfWidth - overlapPixels,
                halfWidth + overlapPixels
            );
            return allDetections;
        }
        private List<Detection> MergeOverlappingDetections(
            List<Detection> leftDetections,
            List<Detection> rightDetections,
            float overlapStart,
            float overlapEnd)
        {
            var result = new List<Detection>();
            var processedRight = new HashSet<int>();

            foreach (var leftDet in leftDetections)
            {
                bool isDuplicate = false;
                float leftCenter = (leftDet.X1 + leftDet.X2) / 2f;

                if (leftCenter >= overlapStart && leftCenter <= overlapEnd)
                {
                    for (int i = 0; i < rightDetections.Count; i++)
                    {
                        if (processedRight.Contains(i))
                            continue;

                        var rightDet = rightDetections[i];
                        float rightCenter = (rightDet.X1 + rightDet.X2) / 2f;

                        if (rightCenter >= overlapStart && rightCenter <= overlapEnd)
                        {
                            float iou = prePro.IoU(leftDet, rightDet);

                            if (iou > 0.5f)
                            {
                                if (rightDet.Score > leftDet.Score)
                                {
                                    result.Add(rightDet);
                                    isDuplicate = true;
                                }
                                else
                                {
                                    result.Add(leftDet);
                                }

                                processedRight.Add(i);
                                isDuplicate = true;
                                break;
                            }
                        }
                    }
                }

                if (!isDuplicate)
                {
                    result.Add(leftDet);
                }
            }

            for (int i = 0; i < rightDetections.Count; i++)
            {
                if (!processedRight.Contains(i))
                {
                    result.Add(rightDetections[i]);
                }
            }

            return result;
        }
    }
}
