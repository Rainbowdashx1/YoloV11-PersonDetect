namespace YoloPerson.VideoSources
{
    internal enum VideoSourceType
    {
        File,           // Archivo local con OpenCvSharp
        RtspOpenCV,     // RTSP con OpenCvSharp
        RtspFFmpeg,     // RTSP con FFmpeg (baja latencia)
        MjpegOpenCV,    // MJPEG con OpenCvSharp
        MjpegFFmpeg     // MJPEG con FFmpeg
    }

    internal static class VideoSourceFactory
    {
        public static IVideoSource Create(string path, VideoSourceType? preferredType = null, bool lowLatency = true)
        {
            VideoSourceType sourceType;

            if (preferredType.HasValue)
            {
                sourceType = preferredType.Value;
            }
            else
            {
                sourceType = DetectSourceType(path);
            }

            Console.WriteLine($"[VideoSourceFactory] Creando fuente tipo: {sourceType}");

            return sourceType switch
            {
                VideoSourceType.File => new FileVideoSource(path),
                VideoSourceType.RtspOpenCV => new RtspOpenCvSource(path, lowLatency),
                VideoSourceType.RtspFFmpeg => new RtspFFmpegSource(path, lowLatency: lowLatency),
                VideoSourceType.MjpegOpenCV => new RtspOpenCvSource(path, lowLatency),
                VideoSourceType.MjpegFFmpeg => new RtspFFmpegSource(path, lowLatency: lowLatency),
                _ => throw new ArgumentException($"Tipo de fuente no soportado: {sourceType}")
            };
        }

        private static VideoSourceType DetectSourceType(string path)
        {
            if (path.StartsWith("rtsp://", StringComparison.OrdinalIgnoreCase))
            {
                return VideoSourceType.RtspFFmpeg;
            }
            else if (path.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                     path.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                if (path.Contains("mjpeg", StringComparison.OrdinalIgnoreCase) ||
                    path.EndsWith(".mjpg", StringComparison.OrdinalIgnoreCase))
                {
                    return VideoSourceType.MjpegFFmpeg;
                }
                return VideoSourceType.RtspOpenCV;
            }
            else
            {
                return VideoSourceType.File;
            }
        }

        public static string GetSourceDescription(VideoSourceType type)
        {
            return type switch
            {
                VideoSourceType.File => "Archivo de video local",
                VideoSourceType.RtspOpenCV => "RTSP con OpenCvSharp (compatible)",
                VideoSourceType.RtspFFmpeg => "RTSP con FFmpeg (baja latencia)",
                VideoSourceType.MjpegOpenCV => "MJPEG con OpenCvSharp",
                VideoSourceType.MjpegFFmpeg => "MJPEG con FFmpeg (baja latencia)",
                _ => "Desconocido"
            };
        }
    }
}
