/**
 * Remember:
 * 1) Loaded data
 * 2) Created categorical label
 * 3) Implemented generator function for image & label
 * 4) Realized that images in the dataset are not all the same size
 * 5) Normalize input data
 */

import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import { getModel } from "./model";
import BrainTumorData from "./data";

const dataPath = path.join(__dirname, "..", "data");
const trainingPath = path.join(dataPath, "Training");
const testingPath = path.join(dataPath, "Testing");

const DESIRED_IMG_SHAPE = {
  height: 128,
  width: 128,
  channels: 3,
};

const brainTumorData = new BrainTumorData(
  trainingPath,
  testingPath,
  DESIRED_IMG_SHAPE
);

const model = getModel(DESIRED_IMG_SHAPE);

const xs = tf.data.generator(brainTumorData.data("train"));
const ys = tf.data.generator(brainTumorData.labels("train"));
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data
  .zip({ xs, ys })
  .shuffle(100 /* bufferSize */)
  .batch(32)
  .prefetch(32);

const xsTest = tf.data.generator(brainTumorData.data("test"));
const ysTest = tf.data.generator(brainTumorData.labels("test"));
const dsTest = tf.data
  .zip({ xs: xsTest, ys: ysTest })
  .shuffle(100 /* bufferSize */)
  .batch(32);

(async () => {
  // Train the model for 5 epochs.
  await model
    .fitDataset(ds, {
      epochs: 5,
      validationBatchSize: 32,
      validationData: dsTest,
    })
    .then((info) => {
      console.log("Accuracy", info.history.acc);
    });

  await model
    .evaluateDataset(dsTest as tf.data.Dataset<{}>, {})
    .then((info) => {
      console.log((info as tf.Scalar[])[0].dataSync());
      console.log((info as tf.Scalar[])[1].dataSync());
      console.log(model.metricsNames);
    });
})();
