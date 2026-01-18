using OpenCvSharp;

namespace YoloPerson.VideoSources
{
    internal interface IVideoSource : IDisposable
    {
        bool Read(Mat frame);
        int Width { get; }
        int Height { get; }
        double Fps { get; }
        bool IsOpened { get; }
        string SourceType { get; }
    }
}
