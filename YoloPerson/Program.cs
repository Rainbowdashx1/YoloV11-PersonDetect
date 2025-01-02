
using YoloPerson.VideoCapture;

internal class Program
{
    static private string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m.onnx");
    static private string videoPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "PersonasCaminando.mp4");
    static private string videoProcessPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "PersonasCaminando_Process.mp4");
    private static void Main(string[] args)
    {
        Capture Cap = new Capture(videoPath,videoProcessPath,modelPath);
        Cap.run();
    }
}