using OpenCvSharp;

namespace YoloPerson.VideoCapture
{
    public class ProcessFrame
    {
        public Mat Letterbox(Mat src, int dstW, int dstH, out float r, out int padX, out int padY)
        {
            int srcW = src.Width;
            int srcH = src.Height;

            r = Math.Min(dstW / (float)srcW, dstH / (float)srcH);
            int newW = (int)Math.Floor(srcW * r);
            int newH = (int)Math.Floor(srcH * r);

            padX = (dstW - newW) / 2;
            padY = (dstH - newH) / 2;

            Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(newW, newH), interpolation: InterpolationFlags.Linear);
            Mat letterboxMat = new Mat(new OpenCvSharp.Size(dstW, dstH), MatType.CV_8UC3, new Scalar(114, 114, 114));

            var roi = new Rect(padX, padY, newW, newH);
            resized.CopyTo(new Mat(letterboxMat, roi));

            return letterboxMat;
        }
    }
}
