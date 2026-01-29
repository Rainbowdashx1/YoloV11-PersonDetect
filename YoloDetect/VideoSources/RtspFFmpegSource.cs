using OpenCvSharp;
using System.Diagnostics;
using System.Text;

namespace YoloPerson.VideoSources
{
    internal class RtspFFmpegSource : IVideoSource
    {
        private Process? ffmpegProcess;
        private BinaryReader? pipeReader;
        private readonly string url;
        private readonly bool isRtsp;
        private bool isRunning;
        private int frameReadErrors = 0;
        private const int MaxFrameReadErrors = 30;

        public int Width { get; private set; }
        public int Height { get; private set; }
        public double Fps { get; private set; }
        public bool IsOpened => isRunning && frameReadErrors < MaxFrameReadErrors;
        public string SourceType => isRtsp ? "RTSP-FFmpeg" : "MJPEG-FFmpeg";

        public RtspFFmpegSource(string url, int width = 0, int height = 0, double fps = 30, bool lowLatency = true)
        {
            this.url = url;
            this.isRtsp = url.StartsWith("rtsp://", StringComparison.OrdinalIgnoreCase);

            Console.WriteLine($"[{SourceType}] Detectando información del stream...");

            if (width == 0 || height == 0)
            {
                DetectStreamInfo(url, out width, out height, out fps);
            }

            Width = width;
            Height = height;
            Fps = fps;

            StartFFmpegProcess(url, lowLatency);
        }

        private void DetectStreamInfo(string url, out int detectedWidth, out int detectedHeight, out double detectedFps)
        {
            try
            {
                var probeProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "ffprobe",
                        Arguments = $"-v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 \"{url}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                probeProcess.Start();
                string output = probeProcess.StandardOutput.ReadToEnd();
                string error = probeProcess.StandardError.ReadToEnd();
                probeProcess.WaitForExit(5000);

                if (!string.IsNullOrEmpty(error))
                {
                    Console.WriteLine($"[{SourceType}] FFprobe warning: {error}");
                }

                var parts = output.Trim().Split(',');
                
                if (parts.Length >= 3)
                {
                    detectedWidth = int.Parse(parts[0]);
                    detectedHeight = int.Parse(parts[1]);

                    var fpsparts = parts[2].Split('/');
                    detectedFps = double.Parse(fpsparts[0]) / double.Parse(fpsparts[1]);

                    Console.WriteLine($"[{SourceType}] Stream detectado: {detectedWidth}x{detectedHeight} @ {detectedFps:F2} FPS");
                }
                else
                {
                    throw new Exception($"No se pudo detectar información del stream. Output: {output}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{SourceType}] Error detectando stream, usando valores por defecto: {ex.Message}");
                detectedWidth = 1920;
                detectedHeight = 1080;
                detectedFps = 30;
            }
        }

        private void StartFFmpegProcess(string url, bool lowLatency)
        {
            StringBuilder ffmpegArgs = new StringBuilder();

            if (isRtsp)
            {
                ffmpegArgs.Append("-rtsp_transport tcp ");
            }

            if (lowLatency)
            {
                ffmpegArgs.Append("-fflags nobuffer+fastseek+flush_packets ");
                ffmpegArgs.Append("-flags low_delay ");
                ffmpegArgs.Append("-max_delay 0 ");
                ffmpegArgs.Append("-probesize 32 ");
                ffmpegArgs.Append("-analyzeduration 0 ");
            }

            ffmpegArgs.Append($"-i \"{url}\" ");
            ffmpegArgs.Append("-f rawvideo ");
            ffmpegArgs.Append("-pix_fmt bgr24 ");
            ffmpegArgs.Append("-an ");
            ffmpegArgs.Append("-");

            Console.WriteLine($"[{SourceType}] Iniciando FFmpeg con: {ffmpegArgs}");

            ffmpegProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ffmpeg",
                    Arguments = ffmpegArgs.ToString(),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            ffmpegProcess.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data) && e.Data.Contains("error", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine($"[{SourceType}] FFmpeg error: {e.Data}");
                }
            };

            try
            {
                ffmpegProcess.Start();
                ffmpegProcess.BeginErrorReadLine();

                pipeReader = new BinaryReader(ffmpegProcess.StandardOutput.BaseStream);
                isRunning = true;

                Console.WriteLine($"[{SourceType}] Conectado: {Width}x{Height} @ {Fps:F2} FPS (Latencia: {(lowLatency ? "Baja" : "Normal")})");
            }
            catch (Exception ex)
            {
                throw new Exception($"Error iniciando FFmpeg: {ex.Message}", ex);
            }
        }

        public bool Read(Mat frame)
        {
            if (!isRunning || pipeReader == null)
                return false;

            try
            {
                int frameSize = Width * Height * 3;
                byte[] buffer = new byte[frameSize];
                int bytesRead = 0;

                while (bytesRead < frameSize)
                {
                    int read = pipeReader.Read(buffer, bytesRead, frameSize - bytesRead);
                    if (read == 0)
                    {
                        frameReadErrors++;
                        
                        if (frameReadErrors >= MaxFrameReadErrors)
                        {
                            Console.WriteLine($"[{SourceType}] Demasiados errores de lectura ({frameReadErrors}). Desconectando.");
                            isRunning = false;
                        }
                        
                        return false;
                    }
                    bytesRead += read;
                }

                if (frameReadErrors > 0)
                {
                    frameReadErrors = 0;
                }

                unsafe
                {
                    fixed (byte* ptr = buffer)
                    {
                        using Mat temp = Mat.FromPixelData(Height, Width, MatType.CV_8UC3, (IntPtr)ptr);
                        temp.CopyTo(frame);
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                frameReadErrors++;
                
                if (frameReadErrors % 10 == 0)
                {
                    Console.WriteLine($"[{SourceType}] Error leyendo frame ({frameReadErrors}): {ex.Message}");
                }

                return false;
            }
        }

        public void Dispose()
        {
            isRunning = false;

            pipeReader?.Dispose();

            if (ffmpegProcess != null && !ffmpegProcess.HasExited)
            {
                try
                {
                    ffmpegProcess.Kill();
                    ffmpegProcess.WaitForExit(1000);
                }
                catch { }
                
                ffmpegProcess.Dispose();
            }

            Console.WriteLine($"[{SourceType}] Desconectado de {url}");
        }
    }
}
