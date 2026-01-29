using OpenCvSharp;
using YoloPerson.PreProcess;

namespace YoloPerson.VideoCapture
{
    internal class FrameRender
    {
        public void DrawDetections(Mat frame, List<Detection> detections)
        {
            DrawDetectionCounter(frame, detections.Count);

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
        public void DrawDetectionCounter(Mat frame, int count)
        {
            string counterText = $"Detecciones: {count}";
            int fontFace = (int)HersheyFonts.HersheySimplex;
            double fontScale = 1.2;
            int thickness = 2;

            var textSize = Cv2.GetTextSize(counterText, (HersheyFonts)fontFace, fontScale, thickness, out int baseline);

            int padding = 15;
            int boxX = padding;
            int boxY = padding;
            int boxWidth = textSize.Width + padding * 2;
            int boxHeight = textSize.Height + padding * 2;

            using Mat overlay = frame.Clone();

            Cv2.Rectangle(overlay,
                new OpenCvSharp.Point(boxX, boxY),
                new OpenCvSharp.Point(boxX + boxWidth, boxY + boxHeight),
                new Scalar(0, 0, 0), // Negro
                -1); // Relleno

            double alpha = 0.6;
            Cv2.AddWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);

            Cv2.PutText(frame,
                counterText,
                new OpenCvSharp.Point(boxX + padding, boxY + textSize.Height + padding),
                (HersheyFonts)fontFace,
                fontScale,
                new Scalar(0, 255, 0),
                thickness);
        }
    }
}
