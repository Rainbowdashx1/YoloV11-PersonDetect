using OpenCvSharp;

namespace YoloPerson.VideoSources
{
    internal class FileVideoSource : IVideoSource
    {
        private readonly OpenCvSharp.VideoCapture capture;

        public int Width => (int)capture.FrameWidth;
        public int Height => (int)capture.FrameHeight;
        public double Fps => capture.Fps;
        public bool IsOpened => capture.IsOpened();
        public string SourceType => "File";

        public FileVideoSource(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"El archivo de video no existe: {filePath}");
            }

            capture = new OpenCvSharp.VideoCapture(filePath);
            
            if (!capture.IsOpened())
            {
                throw new Exception($"No se pudo abrir el archivo de video: {filePath}");
            }

            Console.WriteLine($"[FileVideoSource] Archivo abierto: {Path.GetFileName(filePath)}");
            Console.WriteLine($"[FileVideoSource] Resolución: {Width}x{Height} @ {Fps:F2} FPS");
        }

        public bool Read(Mat frame)
        {
            return capture.Read(frame) && !frame.Empty();
        }

        public void Dispose()
        {
            capture?.Dispose();
            Console.WriteLine("[FileVideoSource] Recurso liberado");
        }
    }
}
