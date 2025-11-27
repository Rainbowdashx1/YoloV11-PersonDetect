
using YoloPerson.VideoCapture;

internal class Program
{
    static private string yolo11m = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m.onnx");
    static private string yolo11m2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m2batch.onnx");
    static private string yolo11n1batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");
    static private string yolo11n2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n2batch.onnx");
    static bool batch = false;

    static private string modelPath;
    static private string videoPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking.mp4");
    static private string videoProcessPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking_Processv3.mp4");
    private static void Main(string[] args)
    {
        Console.WriteLine("=== YoloPerson Detection ===");
        Console.WriteLine("1. Procesar usando yolo11m 1 batch");
        Console.WriteLine("2. Procesar usando yolo11m 2 batch - two batch");
        Console.WriteLine("3. Procesar usando yolo11n 1 batch");
        Console.WriteLine("4. Procesar usando yolo11n 2 batch - two batch");
        Console.WriteLine("5. Salir");
        Console.Write("\nSelecciona una opción: ");

        string? opcion = Console.ReadLine();

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
                Console.WriteLine("Saliendo...");
                return;
            default:
                Console.WriteLine("Opción no válida");
                break;
        }

        Capture Cap = new Capture(videoPath,videoProcessPath,modelPath);

        if(batch)
            Cap.runWithModel2Batch();
        else
            Cap.runWithModel1Batch();
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
        Console.WriteLine("Usando modelo yolo11n 2 batch - two eye");
    }
}