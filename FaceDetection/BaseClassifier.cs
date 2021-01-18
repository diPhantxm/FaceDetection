using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FaceDetection
{
    [Serializable]
    public abstract class BaseClassifier
    {
        public double Threshold { get; protected set; }
        public double Parity { get; protected set; }

        protected void DetermineThreshold(List<Tuple<double, bool, double>> scores)
        {
            var TPos = scores.Where(s => s.Item2).Sum(s => s.Item1);
            var TNeg = scores.Where(s => !s.Item2).Sum(s => s.Item1);

            var minError = double.MaxValue;
            var wPosBelow = 0.0;
            var wNegBelow = 0.0;

            for (int i = 0; i < scores.Count; i++)
            {
                var score = scores[i];

                //var wPosBelow = scores.Where(s => s.Item2 && s.Item1 < score.Item1).Sum(s => s.Item3);
                //var wNegBelow = scores.Where(s => !s.Item2 && s.Item1 < score.Item1).Sum(s => s.Item3);

                if (score.Item2) wPosBelow += score.Item3;
                else wNegBelow += score.Item3;

                var before = wPosBelow + TNeg - wNegBelow;
                var after = wNegBelow + TPos - wPosBelow;

                if (before < after)
                {
                    if (before < minError)
                    {
                        minError = before;
                        Threshold = score.Item1;
                        Parity = -1;
                    }
                }
                else
                {
                    if (after < minError)
                    {
                        minError = after;
                        Threshold = score.Item1;
                        Parity = 1;
                    }
                }
            }
        }
    }
}
