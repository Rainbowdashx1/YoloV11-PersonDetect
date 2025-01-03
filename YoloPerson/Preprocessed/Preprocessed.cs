﻿using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloPerson.PreProcess
{
    public class Preprocessed
    {
        public List<Detection> PreproccessedOutput(Tensor<float>? output0, int padX, int padY, float r, bool nonMaxSuppression = true, float nonMaxSuppressionThreshold = 0.45f, float thresHold = 0.25f)
        {
            List<Detection> detections = new List<Detection>();

            if (output0 is null)
                return detections;

            var dims = output0.Dimensions;
            int batch = dims[0];
            int channels = dims[1];
            int numPreds = dims[2];
            int maxClsIdx = 4;

            for (int i = 0; i < numPreds; i++)
            {
                float xCenter = output0[0, 0, i];
                float yCenter = output0[0, 1, i];
                float w = output0[0, 2, i];
                float h = output0[0, 3, i];

                float clsScore = output0[0, 4, i]; // We only look for the score of people; if you want to search for something else, change the 4 or modify the search logic

                if (clsScore < thresHold)
                    continue;

                // --------------------------------------------------------------------------------
                // A) xywh -> x1, y1, x2, y2 in the LETTERBOX IMAGE (640x640)
                // --------------------------------------------------------------------------------
                float x1_640 = xCenter - w / 2f;
                float y1_640 = yCenter - h / 2f;
                float x2_640 = xCenter + w / 2f;
                float y2_640 = yCenter + h / 2f;

                // --------------------------------------------------------------------------------
                // B) Remove the padding applied in the letterbox
                // --------------------------------------------------------------------------------
                float x1_nopad = x1_640 - padX;
                float y1_nopad = y1_640 - padY;
                float x2_nopad = x2_640 - padX;
                float y2_nopad = y2_640 - padY;

                // --------------------------------------------------------------------------------
                // C) Scale back to the original image by dividing by 'ratio'
                // --------------------------------------------------------------------------------
                float x1_orig = x1_nopad / r;
                float y1_orig = y1_nopad / r;
                float x2_orig = x2_nopad / r;
                float y2_orig = y2_nopad / r;

                // Store detection in the ORIGINAL image coordinates
                detections.Add(new Detection(
                    x1_orig,
                    y1_orig,
                    x2_orig,
                    y2_orig,
                    clsScore,
                    maxClsIdx
                ));
            }

            if (nonMaxSuppression)
            {
                var finalDetections = NonMaxSuppression(detections, nonMaxSuppressionThreshold);
                return finalDetections;
            }
            else
            {
                return detections;
            }
        }
        private List<Detection> NonMaxSuppression(List<Detection> detections, float iouThreshold)
        {
            var sorted = new List<Detection>(detections);
            sorted.Sort((a, b) => b.Score.CompareTo(a.Score));

            var final = new List<Detection>();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                final.Add(best);
                sorted.RemoveAt(0);

                sorted.RemoveAll(det => IoU(best, det) > iouThreshold);
            }
            return final;
        }
        private float IoU(Detection a, Detection b)
        {
            float interX1 = Math.Max(a.X1, b.X1);
            float interY1 = Math.Max(a.Y1, b.Y1);
            float interX2 = Math.Min(a.X2, b.X2);
            float interY2 = Math.Min(a.Y2, b.Y2);

            float interW = Math.Max(0, interX2 - interX1);
            float interH = Math.Max(0, interY2 - interY1);
            float interArea = interW * interH;

            float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
            float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);

            float iou = interArea / (areaA + areaB - interArea);
            return iou;
        }
    }
    public struct Detection
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public float Score;
        public int ClassId;
        public Detection(float x1, float y1, float x2, float y2, float score, int classId)
        {
            X1 = x1; Y1 = y1; X2 = x2; Y2 = y2;
            Score = score;
            ClassId = classId;
        }
    }
}
