using System.Diagnostics;
using System.Runtime.InteropServices;

namespace YoloPerson.VideoSources
{
    internal static class FFmpegHelper
    {
        private static string? ffmpegPath;
        private static bool isInitialized = false;

        public static string FFmpegPath
        {
            get
            {
                EnsureInitialized();
                if (ffmpegPath == null)
                {
                    throw new FileNotFoundException("Error en ffmpegPath");
                }
                return ffmpegPath;
            }
        }
        public static bool IsAvailable
        {
            get
            {
                EnsureInitialized();
                return ffmpegPath != null;
            }
        }

        private static void EnsureInitialized()
        {
            if (!isInitialized)
            {
                Initialize();
            }
        }

        public static void Initialize(string? customFFmpegPath = null, string? customFFprobePath = null)
        {
            isInitialized = true;

            // PASO 1: Intentar extraer FFmpeg del .rar si es necesario
            if (!FFmpegInstaller.AreFilesExtracted())
            {
                FFmpegInstaller.ExtractFFmpeg();
            }

            // PASO 2: Usar rutas personalizadas si se proporcionan
            if (!string.IsNullOrEmpty(customFFmpegPath) && File.Exists(customFFmpegPath))
            {
                ffmpegPath = customFFmpegPath;
            }

            // PASO 3: Buscar en carpeta de ejecución (donde se extrajo el .rar)
            if (ffmpegPath == null)
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                
                if (ffmpegPath == null)
                {
                    string localFFmpeg = Path.Combine(baseDir, "ffmpeg.exe");
                    if (File.Exists(localFFmpeg))
                    {
                        ffmpegPath = localFFmpeg;
                    }
                }
            }

            // PASO 4: Buscar en PATH del sistema
            if (ffmpegPath == null)
            {
                ffmpegPath = FindExecutableInPath("ffmpeg");
                if (ffmpegPath != null)
                {
                    Console.WriteLine($"[FFmpegHelper] FFmpeg encontrado (PATH): {ffmpegPath}");
                }
            }

            // Verificar versiones
            if (ffmpegPath != null)
            {
                VerifyFFmpegVersion(ffmpegPath);
            }

            if (ffmpegPath == null)
            {
                Console.WriteLine("[FFmpegHelper] FFmpeg no está disponible");
            }
        }

        private static string? FindExecutableInPath(string executableName)
        {
            string extension = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ".exe" : "";
            string fileName = executableName + extension;

            var pathVariable = Environment.GetEnvironmentVariable("PATH");
            if (pathVariable == null)
                return null;

            var paths = pathVariable.Split(Path.PathSeparator);

            foreach (var path in paths)
            {
                try
                {
                    string fullPath = Path.Combine(path, fileName);
                    if (File.Exists(fullPath))
                    {
                        return fullPath;
                    }
                }
                catch
                {
                    // Ignorar errores de acceso a directorios
                }
            }

            return null;
        }

        private static void VerifyFFmpegVersion(string ffmpegPath)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = ffmpegPath,
                        Arguments = "-version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                string output = process.StandardOutput.ReadLine() ?? "";
                process.WaitForExit(1000);
            }
            catch
            {
                // No crítico si falla la verificación
            }
        }

        public static bool TestConnection(string testUrl = "")
        {
            if (!IsAvailable)
                return false;

            if (string.IsNullOrEmpty(testUrl))
                return true;

            try
            {            
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = FFmpegPath,
                        Arguments = $"-i \"{testUrl}\" -t 1 -f null -",
                        UseShellExecute = false,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                bool exited = process.WaitForExit(10000);
                
                if (!exited)
                {
                    process.Kill();
                    return false;
                }

                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        public static void ForceReinstall()
        {
            FFmpegInstaller.CleanupExtractedFiles();
            isInitialized = false;
            ffmpegPath = null;
            Initialize();
        }
    }
}
