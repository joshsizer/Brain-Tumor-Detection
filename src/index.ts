/**
 * Remember:
 * 1) Loaded data
 * 2) Created categorical label
 * 3) Implemented generator function for image & label
 * 4) Realized that images in the dataset are not all the same size
 * 5) Normalize input data
 */

import * as tf from "@tensorflow/tfjs-node";
import { getModel } from "./model";
import { data, labels } from "./data";

const IMG_HEIGHT = 128;
const IMG_WIDTH = 128;
const IMG_CHANNELS = 3;

const model = getModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS);

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({ xs, ys }).shuffle(100 /* bufferSize */).batch(32);
ds;

// Train the model for 5 epochs.
model.fitDataset(ds, { epochs: 10 }).then((info) => {
  console.log("Accuracy", info.history.acc);
});
