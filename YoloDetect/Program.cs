

using System.Diagnostics;
using YoloPerson.VideoCapture;
using YoloPerson.VideoSources;

internal class Program
{
    static private string yolo11m = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m.onnx");
    static private string yolo11m2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m2batch.onnx");
    static private string yolo11n1batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");
    static private string yolo11n2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n2batch.onnx");
    static private string yolo26n1batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo26n1batch.onnx");
    static private string yolo26n2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo26n2batch.onnx");

    static bool batch = false;

    static private string modelPath = string.Empty;
    static private string videoPath = string.Empty;
    static private string? videoProcessPath;
    static private VideoSourceType? sourceType;

    private static void Main(string[] args)
    {
        Console.WriteLine("=== YoloPerson Detection ===\n");
        // Se descomprimira Ffmpeg
        FFmpegHelper.Initialize();
        
        if (FFmpegHelper.IsAvailable)
        {
            Console.WriteLine("FFmpeg disponible\n");
        }
        else
        {
            Console.WriteLine("FFmpeg NO disponible (las opciones 3 y 5 no funcionarán)\n");
        }

        Console.WriteLine("=== Selecciona fuente de video ===");
        Console.WriteLine("1. Archivo local (people-walking.mp4)");
        Console.WriteLine($"2. Stream RTSP (con OpenCvSharp)");
        Console.WriteLine($"3. Stream RTSP (con FFmpeg - baja latencia) {(FFmpegHelper.IsAvailable ? "" : "[NO DISPONIBLE]")}");
        Console.WriteLine($"4. Stream MJPEG (con OpenCvSharp)");
        Console.WriteLine($"5. Stream MJPEG (con FFmpeg - baja latencia) {(FFmpegHelper.IsAvailable ? "" : "[NO DISPONIBLE]")}");
        Console.Write("\nOpción: ");

        string? sourceOption = Console.ReadLine();

        switch (sourceOption)
        {
            case "1":
                videoPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking.mp4");
                videoProcessPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking_Processv3.mp4");
                sourceType = VideoSourceType.File;
                break;
            case "2":
                Console.Write("Ingresa URL RTSP (ej: rtsp://192.168.1.100:554/stream): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.RtspOpenCV;
                break;
            case "3":
                if (!FFmpegHelper.IsAvailable)
                {
                    Console.WriteLine("\nFFmpeg no está disponible. Usa la opción 2 en su lugar.");
                    return;
                }
                Console.Write("Ingresa URL RTSP (ej: rtsp://192.168.1.100:554/stream): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.RtspFFmpeg;
                break;
            case "4":
                Console.Write("Ingresa URL MJPEG (ej: http://192.168.1.100/video): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.MjpegOpenCV;
                break;
            case "5":
                if (!FFmpegHelper.IsAvailable)
                {
                    Console.WriteLine("\nFFmpeg no está disponible. Usa la opción 4 en su lugar.");
                    return;
                }
                Console.Write("Ingresa URL MJPEG (ej: http://192.168.1.100/video): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.MjpegFFmpeg;
                break;
            default:
                Console.WriteLine("Opción no válida");
                return;
        }

        Console.WriteLine($"\n=== Fuente seleccionada: {VideoSourceFactory.GetSourceDescription(sourceType.Value)} ===");
        Console.WriteLine("\n=== Selecciona modelo ===");
        Console.WriteLine("1. Procesar usando yolo11m 1 batch");
        Console.WriteLine("2. Procesar usando yolo11m 2 batch - two batch");
        Console.WriteLine("3. Procesar usando yolo11n 1 batch");
        Console.WriteLine("4. Procesar usando yolo11n 2 batch - two batch");
        Console.WriteLine("5. Procesar usando yolo26n 1 batch");
        Console.WriteLine("6. Procesar usando yolo26n 2 batch - two batch");
        Console.WriteLine("7. Salir");
        Console.Write("\nSelecciona una opción: ");

        string? opcion = Console.ReadLine();
        bool Yolo26 = false;
        switch (opcion)
        {
            case "1":
                usingYolo11m();
                break;
            case "2":
                usingYolo11m2batch();
                break;
            case "3":
                usingYolo11n1batch();
                break;
            case "4":
                usingYolo11n2batch();
                break;
            case "5":
                usingYolo26n1batch();
                Yolo26 = true;
                break;
            case "6":
                usingYolo26n2batch();
                Yolo26 = true;
                break;
            case "7":
                Console.WriteLine("Saliendo...");
                return;
            default:
                Console.WriteLine("Opción no válida");
                return;
        }

        Capture Cap = new Capture(videoPath, videoProcessPath, modelPath, sourceType);

        if (batch)
            Cap.runWithModel2Batch();
        else if (!Yolo26)
            Cap.runWithModel1Batch();
        else if (Yolo26)
            Cap.runWithModel1BatchYolo26();
    }
    private static void usingYolo11m()
    {
        modelPath = yolo11m;
        Console.WriteLine("Usando modelo yolo11m");
    }
    private static void usingYolo11m2batch()
    {
        modelPath = yolo11m2batch;
        batch = true;
        Console.WriteLine("Usando modelo yolo11m 2 batch");
    }
    private static void usingYolo11n1batch()
    {
        modelPath = yolo11n1batch;
        Console.WriteLine("Usando modelo yolo11n 1 batch");
    }
    private static void usingYolo11n2batch()
    {
        modelPath = yolo11n2batch;
        batch = true;
        Console.WriteLine("Usando modelo yolo11n 2 batch");
    }
    private static void usingYolo26n1batch()
    {
        modelPath = yolo26n1batch;
        Console.WriteLine("Usando modelo yolo26n 1 batch");
    }
    private static void usingYolo26n2batch()
    {
        modelPath = yolo26n2batch;
        batch = true;
        Console.WriteLine("Usando modelo yolo26n 2 batch");
    }
}