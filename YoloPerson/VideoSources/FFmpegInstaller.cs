using SharpCompress.Archives;
using SharpCompress.Common;

namespace YoloPerson.VideoSources
{
    internal static class FFmpegInstaller
    {
        private static readonly string BaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string FFmpegSubfolder = Path.Combine(BaseDirectory, "FFmpeg");
        private static readonly string FFmpegRarPath = Path.Combine(FFmpegSubfolder, "ffmpeg.rar");
        private static readonly string FFprobeRarPath = Path.Combine(FFmpegSubfolder, "ffprobe.rar");
        
        private static readonly string[] RequiredFiles = { "ffmpeg.exe", "ffprobe.exe" };

        public static bool AreFilesExtracted()
        {
            return RequiredFiles.All(file => File.Exists(Path.Combine(BaseDirectory, file)));
        }

        public static bool ExtractFFmpeg()
        {
            try
            {
                // Verificar si ya están extraídos
                if (AreFilesExtracted())
                {
                    return true;
                }

                bool ffmpegExtracted = false;
                bool ffprobeExtracted = false;

                // Extraer ffmpeg.rar
                if (File.Exists(FFmpegRarPath))
                {
                    ffmpegExtracted = ExtractRarFile(FFmpegRarPath, "ffmpeg.exe");
                }

                // Extraer ffprobe.rar
                if (File.Exists(FFprobeRarPath))
                {
                    ffprobeExtracted = ExtractRarFile(FFprobeRarPath, "ffprobe.exe");
                }

                // Retornar true si se extrajeron ambos o al menos uno
                return ffmpegExtracted || ffprobeExtracted;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FFmpegInstaller] Error: {ex.Message}");
                return false;
            }
        }

        private static bool ExtractRarFile(string rarPath, string targetFile)
        {
            try
            {
                // Crear directorio temporal
                string tempExtractPath = Path.Combine(Path.GetTempPath(), $"ffmpeg_extract_{Guid.NewGuid()}");
                Directory.CreateDirectory(tempExtractPath);

                try
                {
                    // Extraer el archivo .rar
                    using (var archive = ArchiveFactory.Open(rarPath))
                    {
                        foreach (var entry in archive.Entries.Where(e => !e.IsDirectory))
                        {
                            entry.WriteToDirectory(tempExtractPath, new ExtractionOptions()
                            {
                                ExtractFullPath = true,
                                Overwrite = true
                            });
                        }
                    }

                    // Buscar y copiar el archivo objetivo a la raíz
                    string? foundPath = FindFile(tempExtractPath, targetFile);
                    
                    if (foundPath != null)
                    {
                        string destinationPath = Path.Combine(BaseDirectory, targetFile);
                        File.Copy(foundPath, destinationPath, true);
                        Console.WriteLine($"[FFmpegInstaller] {targetFile} extraído exitosamente");
                    }

                    // Limpiar directorio temporal
                    try
                    {
                        Directory.Delete(tempExtractPath, true);
                    }
                    catch
                    {
                        // No crítico si falla la limpieza
                    }

                    return foundPath != null;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[FFmpegInstaller] Error durante la extracción de {targetFile}: {ex.Message}");
                    // Limpiar en caso de error
                    try
                    {
                        if (Directory.Exists(tempExtractPath))
                        {
                            Directory.Delete(tempExtractPath, true);
                        }
                    }
                    catch { }
                    
                    return false;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FFmpegInstaller] Error extrayendo {targetFile}: {ex.Message}");
                return false;
            }
        }

        private static string? FindFile(string directory, string fileName)
        {
            try
            {
                // Buscar en el directorio actual
                string directPath = Path.Combine(directory, fileName);
                if (File.Exists(directPath))
                    return directPath;

                // Buscar recursivamente en subdirectorios
                var files = Directory.GetFiles(directory, fileName, SearchOption.AllDirectories);
                return files.Length > 0 ? files[0] : null;
            }
            catch
            {
                return null;
            }
        }

        public static void CleanupExtractedFiles()
        {
            try
            {
                foreach (string file in RequiredFiles)
                {
                    string filePath = Path.Combine(BaseDirectory, file);
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FFmpegInstaller] Error limpiando archivos: {ex.Message}");
            }
        }

        public static long GetRarFileSize()
        {
            try
            {
                if (File.Exists(FFmpegRarPath))
                {
                    var fileInfo = new FileInfo(FFmpegRarPath);
                    return fileInfo.Length;
                }
            }
            catch { }
            
            return 0;
        }

        public static string GetRarFileSizeFormatted()
        {
            long bytes = GetRarFileSize();
            if (bytes == 0)
                return "N/A";

            string[] sizes = { "B", "KB", "MB", "GB" };
            int order = 0;
            double size = bytes;

            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }

            return $"{size:0.##} {sizes[order]}";
        }
    }
}
