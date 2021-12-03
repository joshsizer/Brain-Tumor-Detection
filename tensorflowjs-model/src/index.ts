/**
 * Remember:
 * 1) Loaded data
 * 2) Created categorical label
 * 3) Implemented generator function for image & label
 * 4) Realized that images in the dataset are not all the same size
 * 5) Normalize input data
 */

/**
 * Detect brain tumors using a convolutional neural network.
 * Can classify as having "no tumor," "glioma," "meningioma,"
 * or "pituitary." The data set for this project comes from
 * the Kaggle "Brain Tumor MRI Dataset," uploaded by user
 * "Masoud Nickparavar."
 *
 * https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset
 */
import * as tf from "@tensorflow/tfjs-node";
import BrainTumorModel from "./model";
import BrainTumorData from "./data";
import imagePreprocessor from "./preprocessor";
import path from "path";

// Define our data paths here, so it is clear where the data
// is coming from. I've used path.join() to keep this platform
// independent.
const DATA_PATH = path.join(__dirname, "..", "data");
const TRAINING_PATH = path.join(DATA_PATH, "Training");
const TESTING_PATH = path.join(DATA_PATH, "Testing");
const SAVE_MODEL_PATH = path.join(__dirname, "..", "model");

// The preprocessor, model, and data generators need to know
// the desired image shape. Ideally, the data generators should
// not be dependant on the desired image shape; however, the
// easiest way to coerce the input images to the correct number
// of channels is via tf.node.decodeImage, which optionally
// accepts the desired number of channels for the returned tensor.
// While there is a tf.image.grayscaleToRGB function, there
// is not a tf.image.RGBToGrayscale function at this moment.
// The lack of ability to switch from 3 channels (RGB) to 1
// channel (grayscale) means that the easiest place to define
// the number of channels in our processed images is, unfortunately,
// in the readImage function used by the data generators.
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

// Pass in the desired image shape to allow for quick changes
// in the model input shape. Helps for testing different model
// architectures as well as tuning model performance.
const model = new BrainTumorModel(DESIRED_IMG_SHAPE);

const NUM_EPOCHS = 40;
const BATCH_SIZE = 32;

// Tensorflow is streaming images from disk instead of loading
// them entirely into memory, so the shuffle process for shuffling
// a batch of data must occur over a sample size larger than
// the batch size.
const SHUFFLE_BUFFER_SIZE = 100;

// While a typical machine could load the entire dataset into
// memory (~250 Mb), I opted to use stream generator functions
// passed to tf.data.generator in case the dataset ever grows
// in size.
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
      console.log("Loss", info.history.loss);
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
