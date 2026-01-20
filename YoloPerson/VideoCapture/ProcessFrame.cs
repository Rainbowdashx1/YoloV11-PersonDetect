using OpenCvSharp;

namespace YoloPerson.VideoCapture
{
    public class ProcessFrame
    {
        // Buffer reutilizable para evitar allocaciones repetidas (usado en método optimizado)
        private Mat? _resizedBuffer;
        private Size _lastResizedSize;

        /// <summary>
        /// Método original de Letterbox
        /// </summary>
        public void Letterbox(Mat src, Mat dst, int dstW, int dstH, out float r, out int padX, out int padY)
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
            
            // Llenar dst con color de fondo
            dst.SetTo(new Scalar(114, 114, 114));
            
            var roi = new Rect(padX, padY, newW, newH);
            resized.CopyTo(new Mat(dst, roi));
            
            resized.Dispose();
        }

        /// <summary>
        /// Letterbox optimizado usando CopyMakeBorder
        /// </summary>
        public void LetterboxOptimized(Mat src, Mat dst, int dstW, int dstH, out float r, out int padX, out int padY)
        {
            int srcW = src.Width;
            int srcH = src.Height;

            float rW = dstW / (float)srcW;
            float rH = dstH / (float)srcH;
            r = rW < rH ? rW : rH;

            int newW = (int)(srcW * r);
            int newH = (int)(srcH * r);

            int totalPadX = dstW - newW;
            int totalPadY = dstH - newH;
            padX = totalPadX >> 1;
            padY = totalPadY >> 1;
            int padRight = totalPadX - padX;
            int padBottom = totalPadY - padY;

            Size newSize = new Size(newW, newH);
            if (_resizedBuffer == null || _lastResizedSize != newSize)
            {
                _resizedBuffer?.Dispose();
                _resizedBuffer = new Mat();
                _lastResizedSize = newSize;
            }

            Cv2.Resize(src, _resizedBuffer, newSize, 0, 0, InterpolationFlags.Linear);
            Cv2.CopyMakeBorder(
                _resizedBuffer, 
                dst, 
                padY, padBottom, 
                padX, padRight, 
                BorderTypes.Constant, 
                new Scalar(114, 114, 114)
            );
        }

        /// <summary>
        /// Liberar recursos del buffer
        /// </summary>
        public void DisposeBuffers()
        {
            _resizedBuffer?.Dispose();
            _resizedBuffer = null;
        }
    }
}
