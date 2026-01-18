using SharpCompress.Archives;
using SharpCompress.Common;

namespace YoloPerson.VideoSources
{
    internal static class FFmpegInstaller
    {
        private static readonly string BaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string FFmpegSubfolder = Path.Combine(BaseDirectory, "FFmpeg");
        private static readonly string FFmpegRarPath = Path.Combine(FFmpegSubfolder, "ffmpeg.rar");
        
        private static readonly string[] RequiredFiles = { "ffmpeg.exe" };

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

                // Verificar si existe el archivo .rar
                if (!File.Exists(FFmpegRarPath))
                {
                    return false;
                }

                // Crear directorio temporal
                string tempExtractPath = Path.Combine(Path.GetTempPath(), $"ffmpeg_extract_{Guid.NewGuid()}");
                Directory.CreateDirectory(tempExtractPath);

                try
                {
                    // Extraer el archivo .rar
                    using (var archive = ArchiveFactory.Open(FFmpegRarPath))
                    {
                        int totalFiles = archive.Entries.Count();
                        int extractedFiles = 0;

                        foreach (var entry in archive.Entries.Where(e => !e.IsDirectory))
                        {
                            entry.WriteToDirectory(tempExtractPath, new ExtractionOptions()
                            {
                                ExtractFullPath = true,
                                Overwrite = true
                            });

                            extractedFiles++;
                        }
                    }

                    // Buscar y copiar ffmpeg.exe y ffprobe.exe a la raíz
                    bool foundAll = true;
                    foreach (string requiredFile in RequiredFiles)
                    {
                        string? foundPath = FindFile(tempExtractPath, requiredFile);
                        
                        if (foundPath != null)
                        {
                            string destinationPath = Path.Combine(BaseDirectory, requiredFile);
                            File.Copy(foundPath, destinationPath, true);
                        }
                        else
                        {
                            foundAll = false;
                        }
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

                    if (foundAll)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[FFmpegInstaller] Error durante la extracción: {ex.Message}");
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
                Console.WriteLine($"[FFmpegInstaller] Error: {ex.Message}");
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
