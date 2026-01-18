using OpenCvSharp;

namespace YoloPerson.VideoSources
{
    internal class RtspOpenCvSource : IVideoSource
    {
        private readonly OpenCvSharp.VideoCapture capture;
        private readonly string url;
        private readonly bool isRtsp;
        private int consecutiveFailures = 0;
        private const int MaxConsecutiveFailures = 30;

        public int Width { get; private set; }
        public int Height { get; private set; }
        public double Fps { get; private set; }
        public bool IsOpened => capture.IsOpened() && consecutiveFailures < MaxConsecutiveFailures;
        public string SourceType => isRtsp ? "RTSP-OpenCV" : "MJPEG-OpenCV";

        public RtspOpenCvSource(string url, bool lowLatency = true)
        {
            this.url = url;
            this.isRtsp = url.StartsWith("rtsp://", StringComparison.OrdinalIgnoreCase);

            Console.WriteLine($"[{SourceType}] Conectando a: {url}");

            capture = new OpenCvSharp.VideoCapture(url, VideoCaptureAPIs.FFMPEG);

            if (!capture.IsOpened())
            {
                throw new Exception($"No se pudo conectar a {url}");
            }

            Width = (int)capture.FrameWidth;
            Height = (int)capture.FrameHeight;
            Fps = capture.Fps > 0 ? capture.Fps : 30;

            if (lowLatency)
            {
                capture.Set(VideoCaptureProperties.BufferSize, 1);
                Console.WriteLine($"[{SourceType}] Modo baja latencia activado (buffer=1)");
            }
            else
            {
                capture.Set(VideoCaptureProperties.BufferSize, 3);
            }

            capture.Set(VideoCaptureProperties.FourCC, FourCC.H264);

            Console.WriteLine($"[{SourceType}] Conectado: {Width}x{Height} @ {Fps:F2} FPS");
        }

        public bool Read(Mat frame)
        {
            if (!capture.IsOpened())
            {
                return false;
            }

            bool success = capture.Read(frame);

            if (!success || frame.Empty())
            {
                consecutiveFailures++;
                
                if (consecutiveFailures >= MaxConsecutiveFailures)
                {
                    Console.WriteLine($"[{SourceType}] Demasiados fallos consecutivos ({consecutiveFailures}). Desconectando.");
                    return false;
                }

                if (consecutiveFailures % 10 == 0)
                {
                    Console.WriteLine($"[{SourceType}] Advertencia: {consecutiveFailures} fallos consecutivos");
                }

                return false;
            }

            if (consecutiveFailures > 0)
            {
                Console.WriteLine($"[{SourceType}] Recuperado después de {consecutiveFailures} fallos");
                consecutiveFailures = 0;
            }

            return true;
        }

        public void Dispose()
        {
            capture?.Dispose();
            Console.WriteLine($"[{SourceType}] Desconectado de {url}");
        }
    }
}
