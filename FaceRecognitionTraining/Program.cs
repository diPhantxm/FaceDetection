using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Numpy;
using Keras.Models;
using Keras.Layers;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.Serialization.Formatters.Binary;

namespace FaceRecognitionTraining
{
    class Program
    {
        static void Main(string[] args)
        {

            var images = np.arange(0).reshape(0, 0, 0, 0);
            var names = new string[0];
            
            LoadData(ref images, ref names);
            var namesND = np.arange(names.Length);
            var possibleOutputs = names.Distinct().ToList();
            for (int i = 0; i < names.Length; i++)
            {
                namesND[i] = (NDarray)possibleOutputs.IndexOf(names[i]);
            }
            var num_classes = possibleOutputs.Count();

            var seq = new Sequential();
            

            // 1 layer
            seq.Add(new Conv2D(32, new Tuple<int, int>(3, 3), activation: "relu", input_shape: new Keras.Shape(250, 250, 1)));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));
            
            // 2 layer
            seq.Add(new Conv2D(64, new Tuple<int, int>(3, 3), activation: "relu"));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));

            // Fully-connected layer
            seq.Add(new Dense(219, activation: "relu"));
            seq.Add(new Dropout(0.2));

            // Output layer
            seq.Add(new Dense(num_classes, activation: "softmax"));

            seq.Compile(new Keras.Optimizers.SGD(lr: 0.001f), "categorical_crossentropy", new string[] { "categorical_accuracy", "categorical_crossentropy" });

            seq.Summary();
            seq.Fit(images, namesND);

            seq.Save("FaceRecognitionNetwork.h5");
        }

        // TODO: get image height and width
        static void LoadData(ref NDarray images, ref string[] names)
        {
            var src = "B:\\Desktop\\New folder\\";

            var nameList = new List<string>();

            var dirs = Directory.GetDirectories(src);
            var imgsCount = dirs.Sum(dir => Directory.GetFiles(dir).Count());

            images = np.arange(imgsCount * 250 * 250).reshape(imgsCount, 250, 250, 1);

            var i = 0;
            foreach (var dir in dirs)
            {
                var fileNames = Directory.GetFiles(dir);
                var name = fileNames[0].Split("\\").Last();

                foreach (var file in fileNames)
                {
                    var image = Image.Load<Rgba32>(file);
                    var grayscale = Grayscale(image);
                    nameList.Add(name);
                    for (int j = 0; j < grayscale.GetLength(0); j++)
                    {
                        for (int k = 0; k < grayscale.GetLength(1); k++)
                        {
                            images[i, j, k, 0] = (NDarray)((double)image[j, k].R + (double)image[j, k].G + (double)image[j, k].B) / 3;
                        }
                    }

                    i++;
                }
            }

            names = nameList.ToArray();
        }

        static double[,] Grayscale(Image<Rgba32> image)
        {
            var result = new double[image.Height, image.Width];

            for (int i = 0; i < image.Height; i++)
            {
                for (int j = 0; j < image.Width; j++)
                {
                    result[i, j] = (double)(image[i, j].R + image[i, j].G + image[i, j].B) / 3;
                }
            }

            return result;
        }
    }
}
