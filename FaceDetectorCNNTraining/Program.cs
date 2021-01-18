using Keras.Layers;
using Keras.Models;
using Numpy;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace FaceDetectorCNNTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            NDarray images = null;
            NDarray results = null;

            results = Keras.Utils.Util.ToCategorical(results, 2);

            var seq = TrainCNN(images, results);
            seq.Summary();

            Console.ReadLine();
        }

        private static Sequential CreateCNN()
        {
            var seq = new Sequential();

            // 1 layer
            seq.Add(new Conv2D(32, new Tuple<int, int>(3, 3), activation: "relu", input_shape: new Keras.Shape(19, 19, 1)));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));

            // 2 layer
            seq.Add(new Conv2D(64, new Tuple<int, int>(3, 3), activation: "relu"));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));

            seq.Add(new Flatten());

            // Fully-connected layer
            seq.Add(new Dense(219, activation: "relu"));

            // Output layer
            seq.Add(new Dense(2, activation: "softmax"));
            

            seq.Compile(new Keras.Optimizers.SGD(lr: 0.001f), "categorical_crossentropy", new string[] { "accuracy" });

            return seq;
        }

        private static Sequential TrainCNN(NDarray images, NDarray results)
        {
            var seq = CreateCNN();

            seq.Summary();
            seq.Fit(images, results, batch_size: 32, epochs: 200);

            seq.SaveWeight("FaceDetectorCNN.h5");
            return seq;
        }

        private static double[,] Grayscale(Image<Rgba32> image)
        {
            var result = new double[image.Height, image.Width];

            for (int i = 0; i < image.Height; i++)
            {
                for (int j = 0; j < image.Width; j++)
                {
                    result[i, j] = (double)(image[j, i].R + image[j, i].G + image[j, i].B) / 3;
                }
            }

            return result;
        }

    }

}
