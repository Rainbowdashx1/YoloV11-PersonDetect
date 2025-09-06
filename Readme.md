# YOLOPerson

## Description

YOLOPerson is a technical project developed in C# .NET Core 8 that utilizes **YOLOv11** for detecting people in videos. This project is designed for frame processing using **Microsoft.ML.OnnxRuntime.Gpu (v1.20.1)** and **OpenCvSharp4**.

## System Used

### Hardware
- **Processor**: Intel Core i7-8700 CPU @ 3.20GHz (Coffee Lake)
  - 6 physical cores, 12 logical cores
- **Operating System**: Windows 10 (22H2/2022 Update)
- **GPU**: NVIDIA GTX1070 8GB

### Software
- **.NET Version**: .NET 8.0.7
- **Microsoft.ML.OnnxRuntime.Gpu**: v1.20.1
- **OpenCvSharp4**
- **CUDA Toolkit**: 12.6 ([download](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows\&target_arch=x86_64\&target_version=10\&target_type=exe_local))
- **cuDNN**: 9.13.0 for CUDA 12.x ([download](https://developer.nvidia.com/cudnn-downloads))

## Setup and Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Rainbowdashx1/YoloV11-PersonDetect
   cd YoloV11-PersonDetect
   ```

2. **Install dependencies**
   Ensure you have installed:
   - .NET Core SDK 8.0.7
   - NVIDIA CUDA Toolkit 12.6
   - cuDNN 8.9.7 compatible with CUDA 12.x

3. **Build the project**

   ```bash
   dotnet build
   ```

4. **Run the project**

   ```bash
   dotnet run
   ```

## Future Improvements

- Add support for real-time detection from a camera.
- Optimize video processing performance.
- Integrate visualization directly into the project interface.
- Incorporate ByteTrack for advanced tracking capabilities.

---

# Changelog

## Version 1.0.1

This version introduces performance optimizations, additional functionality, and improvements in code quality and readability.

### Added
- **`BenchMarksMethods` Project**  
  A new project was added to provide a benchmarking layer, enabling testing of individual methods to determine the most efficient in terms of execution time and CPU usage.
  
- **`MatToTensorParallel` Method in `SessionGpu`**  
  A new method for creating tensors more quickly, with improved execution time in milliseconds compared to the previous method.

### Changed
- **Enhanced `SessionGpu` Constructor**  
  Additional configuration parameters were introduced, resulting in a slight improvement in inference speed.

- **Updated Preprocessing in `PreProcessed.cs`**  
  Adjusted the preprocessing logic to focus exclusively on detecting people, as this is the only required functionality for this implementation. Previously, it iterated over all possible objects YOLOv11 could detect.

- **Comments Updated to English**  
  All comments were reviewed and converted to English for improved clarity and consistency.

### Removed
- **Unnecessary Comments**  
  Redundant or outdated comments were removed for better code readability.

---

## Version 1.0.0
- Initial release with foundational functionality and YOLOv11 integration.

