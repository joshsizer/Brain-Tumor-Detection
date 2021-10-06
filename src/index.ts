import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

const readImage = (path: string) => {
  const imageBuffer = fs.readFileSync(path);
  const tfimage = tf.node.decodeImage(imageBuffer);
  //default #channel 4
  return tfimage;
};

const tfImage = readImage(
  path.join(__dirname, "../data/Training/glioma/Tr-gl_0010.jpg")
);
console.log(tfImage);

const model = tf.sequential();

const IMG_HEIGHT = 512;
const IMG_WIDTH = 512;
const IMG_CHANNELS = 3;

model.add(
  tf.layers.conv2d({
    inputShape: [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

// Repeat another conv2d + maxPooling stack.
// Note that we have more filters in the convolution.
model.add(
  tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling",
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

// Now we flatten the output from the 2D filters into a 1D vector to prepare
// it for input into our last layer. This is common practice when feeding
// higher dimensional data to a final classification output layer.
model.add(tf.layers.flatten());

// Our last layer is a dense layer which has 10 output units, one for each
// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
const NUM_OUTPUT_CLASSES = 4;
model.add(
  tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: "varianceScaling",
    activation: "softmax",
  })
);

// Choose an optimizer, loss function and accuracy metric,
// then compile and return the model
const optimizer = tf.train.adam();
model.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

console.log(model.summary());
