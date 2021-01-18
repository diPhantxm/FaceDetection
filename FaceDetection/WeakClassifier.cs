using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FaceDetection
{
    [Serializable]
    public class WeakClassifier : BaseClassifier, IEquatable<WeakClassifier>
    {
        public HaarFeatureType Type { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public int XPos { get; private set; }
        public int YPos { get; private set; }

        public double Error { get; set; }


        public WeakClassifier(HaarFeatureType type, int height, int width)
        {
            Type = type;
        }

        public WeakClassifier(HaarFeatureType type, int height, int width, int x, int y)
        {
            if (type == HaarFeatureType.Type1 && width % 2 != 0) throw new ArgumentException("Width must be even", nameof(width));
            if (type == HaarFeatureType.Type2 && height % 2 != 0) throw new ArgumentException("Height must be even", nameof(height));
            if (type == HaarFeatureType.Type3 && width % 3 != 0) throw new ArgumentException("Width must be odd and divisible by 3", nameof(width));
            if (type == HaarFeatureType.Type4 && height % 3 != 0) throw new ArgumentException("Height must be odd and divisible by 3", nameof(height));
            if (type == HaarFeatureType.Type5)
            {
                if (width % 2 != 0) throw new ArgumentException("Width must be odd", nameof(width));
                if (height % 2 != 0) throw new ArgumentException("Height must be odd", nameof(height));
            }

            Type = type;
            XPos = x;
            YPos = y;
            Width = width;
            Height = height;

            Parity = 1;
            Threshold = 0.2;
        }       

        public double Apply(double[,] integralImage)
        {
            var white = .0;
            var black = .0;

            if (Type == HaarFeatureType.Type1)
            {
                white = integralImage[YPos + Height, XPos + Width / 2] - integralImage[YPos + Height, XPos] - integralImage[YPos, XPos + Width / 2] + integralImage[YPos, XPos];
                black = integralImage[YPos + Height, XPos + Width] - integralImage[YPos + Height, XPos + Width / 2] - integralImage[YPos, XPos + Width] + integralImage[YPos, XPos];
            }
            if (Type == HaarFeatureType.Type2)
            {
                white = integralImage[YPos + Height / 2, XPos + Width] - integralImage[YPos + Height / 2, XPos] - integralImage[YPos, XPos + Width] + integralImage[YPos, XPos];
                black = integralImage[YPos + Height, XPos + Width] - integralImage[YPos + Height, XPos] - integralImage[YPos + Height / 2, XPos + Width] + integralImage[YPos, XPos];
            }
            if (Type == HaarFeatureType.Type3)
            {
                white = integralImage[YPos + Height, XPos + Width / 3] - integralImage[YPos + Height, XPos] - integralImage[YPos, XPos + Width / 3] + integralImage[YPos, XPos];
                white += integralImage[YPos + Height, XPos + Width] - integralImage[YPos + Height, XPos + 2 * Width / 3] - integralImage[YPos, XPos + 2 * Width / 3] + integralImage[YPos, XPos + 2 * Width / 3];
                black = integralImage[YPos + Height, XPos + 2 * Width / 3] - integralImage[YPos + Height, XPos + Width / 3] - integralImage[YPos, XPos + Width / 3] + integralImage[YPos, XPos + Width / 3];
            }
            if (Type == HaarFeatureType.Type4)
            {
                white = integralImage[YPos + Height / 3, XPos + Width] - integralImage[YPos + Height / 3, XPos] - integralImage[YPos, XPos + Width] + integralImage[YPos, XPos];
                white += integralImage[YPos + Height, XPos + Width] - integralImage[YPos + Height, XPos] - integralImage[YPos + 2 * Height / 3, XPos + Width] + integralImage[YPos + 2 * Height / 3, XPos];
                black = integralImage[YPos + 2 * Height / 3, XPos + Width] - integralImage[YPos + 2 * Height / 3, XPos] - integralImage[YPos + Height / 3, XPos + Width] + integralImage[YPos + Height / 3, XPos];
            }
            if (Type == HaarFeatureType.Type5)
            {
                white = integralImage[YPos + Height, XPos + Width] - integralImage[YPos + Height, XPos + Width / 2] - integralImage[YPos + Height / 2, XPos + Width] + integralImage[YPos + Height / 2, XPos + Width / 2];
                white += integralImage[YPos + Height / 2, XPos + Width / 2] - integralImage[YPos + Height / 2, XPos] - integralImage[YPos, XPos + Width / 2] + integralImage[YPos, XPos];
                black = integralImage[YPos + Height / 2, XPos + Width] - integralImage[YPos + Height / 2, XPos + Width / 2] - integralImage[YPos, XPos + Width] + integralImage[YPos, XPos + Width / 2];
                black += integralImage[YPos + Height, XPos + Width / 2] - integralImage[YPos + Height, XPos] - integralImage[YPos + Height / 2, XPos + Width / 2] + integralImage[YPos + Height / 2, XPos];
            }

            return black - white;
        }

        public bool Detect(double[,] integralImage)
        {
            var score = Apply(integralImage);

            if (score * Parity < Threshold * Parity)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public void Train(ref double[][,] integralImages, ref bool[] results, double[] weights)
        {
            var scores = new List<Tuple<double, bool, double>>();

            // Collecting scores
            for (int i = 0; i < integralImages.Length; i++)
            {
                scores.Add(new Tuple<double, bool, double>(Apply(integralImages[i]), results[i], weights[i]));
            }

            var orderedScores = scores.OrderBy(s => s.Item1).ToList();

            DetermineThreshold(orderedScores);
            Error = .0;

            for (int i = 0; i < integralImages.Length; i++)
            {
                var featureResult = scores[i].Item1 * Parity < Threshold * Parity ? true : false;

                // Feature error
                if (featureResult != results[i])
                {
                    Error += scores[i].Item3;
                }
            }
        }

        public void Reset()
        {
            Threshold = 0.0;
            Parity = 1.0;
        }

        public bool Equals(WeakClassifier other)
        {
            if (Height == other.Height &&
                Width == other.Width &&
                XPos == other.XPos &&
                YPos == other.YPos &&
                Type == other.Type)
            {
                return true;
            }

            return false;
        }
    }


    [Serializable]
    public enum HaarFeatureType
    {
        Type1,
        Type2,
        Type3,
        Type4,
        Type5
    }
}
