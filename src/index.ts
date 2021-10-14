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
import BrainTumorModel from "./model";
import BrainTumorData from "./data";
import imagePreprocessor from "./preprocessor";

const DATA_PATH = path.join(__dirname, "..", "data");
const TRAINING_PATH = path.join(DATA_PATH, "Training");
const TESTING_PATH = path.join(DATA_PATH, "Testing");
const SAVE_MODEL_PATH = path.join(__dirname, "..", "model");

const DESIRED_IMG_SHAPE = {
  height: 128,
  width: 128,
  channels: 3,
};

const brainTumorData = new BrainTumorData(
  TRAINING_PATH,
  TESTING_PATH,
  DESIRED_IMG_SHAPE
);

const model = new BrainTumorModel(DESIRED_IMG_SHAPE);

const NUM_EPOCHS = 1;
const BATCH_SIZE = 32;
const SHUFFLE_BUFFER_SIZE = 100;

const xs = tf.data
  .generator(brainTumorData.data("train"))
  .map(imagePreprocessor(DESIRED_IMG_SHAPE));
const ys = tf.data.generator(brainTumorData.labels("train"));
const ds = tf.data
  .zip({ xs, ys })
  .shuffle(SHUFFLE_BUFFER_SIZE)
  .batch(BATCH_SIZE)
  .prefetch(BATCH_SIZE);

const xsTest = tf.data
  .generator(brainTumorData.data("test"))
  .map(imagePreprocessor(DESIRED_IMG_SHAPE));
const ysTest = tf.data.generator(brainTumorData.labels("test"));
const dsTest = tf.data
  .zip({ xs: xsTest, ys: ysTest })
  .shuffle(SHUFFLE_BUFFER_SIZE)
  .batch(BATCH_SIZE);

(async () => {
  await model
    .fitDataset(ds, {
      epochs: NUM_EPOCHS,
      validationBatchSize: BATCH_SIZE,
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

  await model.save("file://" + SAVE_MODEL_PATH);
})();
