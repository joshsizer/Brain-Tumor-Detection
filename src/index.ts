/**
 * Remember:
 * 1) Loaded data
 * 2) Created categorical label
 * 3) Implemented generator function for image & label
 * 4) Realized that images in the dataset are not all the same size
 * 5) Normalize input data
 */

import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { shuffle } from "./util";

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

const dataPath = path.join(__dirname, "../data");
const trainingPath = path.join(dataPath, "Training");
const testPath = path.join(dataPath, "Testing");
testPath;

const trainingPaths: { path: string; label: string }[] = [];
const allLabels = [];

for (const label of fs.readdirSync(trainingPath)) {
  const labelPath = path.join(trainingPath, label);
  if (!fs.lstatSync(labelPath).isDirectory()) {
    continue;
  }
  allLabels.push(label);
  for (const file of fs.readdirSync(labelPath)) {
    const fullImagePath = path.join(labelPath, file);
    trainingPaths.push({ path: fullImagePath, label });
  }
}

const labelsMap: { [label: string]: number } = {};
for (let i = 0; i < allLabels.length; i++) {
  labelsMap[allLabels[i]] = i;
}

shuffle(trainingPaths);

function labelToCategorical(label: string) {
  const toRet = [];
  for (const l in labelsMap) {
    if (l === label) {
      toRet.push(1.0);
    } else {
      toRet.push(0.0);
    }
  }
  return tf.tensor(toRet);
}

function* data() {
  for (let i = 0; i < trainingPaths.length; i++) {
    // Generate one sample at a time.
    const img = readImage(trainingPaths[i].path);
    console.log(trainingPaths[i]);
    console.log(img);
    yield img;
  }
}

function* labels() {
  for (let i = 0; i < 100; i++) {
    // Generate one sample at a time.
    yield labelToCategorical(trainingPaths[i].label);
  }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({ xs, ys }).shuffle(100 /* bufferSize */).batch(32);
ds;

// Train the model for 5 epochs.
model.fitDataset(ds, { epochs: 5 }).then((info) => {
  console.log("Accuracy", info.history.acc);
});
