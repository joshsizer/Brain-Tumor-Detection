import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { shuffle } from "./util";

/**
 * Encapsulate the brain tumor dataset to allow for easy changes
 * in data sourcing for the future.
 */
export default class BrainTumorData {
  trainingPaths: { path: string; label: string }[];
  testingPaths: { path: string; label: string }[];
  desiredImageShape: { height: number; width: number; channels: number };
  allLabels: string[];
  labelMap: { [label: string]: number };

  /**
   * Initializes data and labels to be streamed to the CNN.
   *
   * @param trainingPath The path to the training dataset.
   * @param testingPath The path to the testing dataset.
   * @param desiredImageShape The input shape to the CNN.
   */
  constructor(
    trainingPath: string,
    testingPath: string,
    desiredImageShape: { height: number; width: number; channels: number }
  ) {
    const { trainingPaths, testingPaths, allLabels } = this.getPathsAndLabels(
      trainingPath,
      testingPath
    );
    this.trainingPaths = trainingPaths;
    this.testingPaths = testingPaths;
    this.desiredImageShape = desiredImageShape;
    this.allLabels = allLabels;
    this.labelMap = this.createLabelMap(allLabels);

    shuffle(this.trainingPaths);
    shuffle(this.testingPaths);
  }

  /**
   * Get a generator function that yields either train or
   * test data.
   *
   * @param which Either "train" or "test" for the respective
   *  data.
   * @returns A generator function for use by
   *  {@link tf.data.generator}.
   */
  data(which: "train" | "test") {
    // "this" is not defined when tensorflow calls
    // the returned function, so we pass along all the
    // dependancies here
    const paths = which === "train" ? this.trainingPaths : this.testingPaths;
    const desiredImageShape = this.desiredImageShape;

    return function* (): Iterator<tf.TensorContainer, any, undefined> {
      for (let i = 0; i < paths.length; i++) {
        yield tf.tidy(() => {
          return readImage(paths[i].path, desiredImageShape.channels);
        });
      }
    };
  }

  /**
   * Get a generator function that yields either train or
   * test labels.
   *
   * @param which Either "train" or "test" for the respective
   *  labels.
   * @returns A generator function for use by
   *  {@link tf.data.generator}.
   */
  labels(which: "train" | "test") {
    const paths = which === "train" ? this.trainingPaths : this.testingPaths;
    const labelToCategorical = this.labelToCategorical;
    const labelMap = this.labelMap;

    return function* (): Iterator<tf.TensorContainer, any, undefined> {
      for (let i = 0; i < paths.length; i++) {
        yield labelToCategorical(paths[i].label, labelMap);
      }
    };
  }

  /**
   * This function collects essential metadata about the image
   * dataset.
   *
   * The "Brain Tumor MRI Dataset," uploaded by user "Masoud
   * Nickparavar," is formatted as such:
   * ```
   * data
   *  |__Testing
   *  |   |_ glioma
   *  |   |   |_ Te-gl_0010.jpg
   *  |   |   |_ ...
   *  |   |_ meningioma
   *  |   |   |_ Te-me_0010.jpg
   *  |   |   |_ ...
   *  |   |_ notumor
   *  |   |   |_ Te-no_0010.jpg
   *  |   |   |_ ...
   *  |   |_ pituitary
   *  |       |_ Te-me_0010.jpg
   *  |       |_ ...
   *  |
   *  |__Training
   *  |   |_ glioma
   *  |   |   |_ Tr-gl_0010.jpg
   *  |   |   |_ ...
   *  |   |_ meningioma
   *  |   |   |_ Tr-me_0010.jpg
   *  |   |   |_ ...
   *  |   |_ notumor
   *  |   |   |_ Tr-no_0010.jpg
   *  |   |   |_ ...
   *  |   |_ pituitary
   *  |       |_ Tr-me_0010.jpg
   *  |       |_ ...
   * ```
   *
   * @param trainingPath The path to the training dataset.
   * @param testingPath The path to the testing dataset.
   * @returns An object containing a list of training image
   *  paths and their labels, a list of testing image paths
   *  and their labels, and a list of labels. In this case,
   *  the labels will be a list like so (not necessarily in
   *  this order):
   *
   *  ```typescript
   *  ["glioma", "meningioma", "notumor", "pituitary"].
   *  ```
   */
  private getPathsAndLabels(
    trainingPath: string,
    testingPath: string
  ): {
    trainingPaths: { path: string; label: string }[];
    testingPaths: { path: string; label: string }[];
    allLabels: string[];
  } {
    const trainingPaths: { path: string; label: string }[] = [];
    const testingPaths: { path: string; label: string }[] = [];
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

    for (const label of fs.readdirSync(testingPath)) {
      const labelPath = path.join(testingPath, label);
      if (!fs.lstatSync(labelPath).isDirectory()) {
        continue;
      }
      for (const file of fs.readdirSync(labelPath)) {
        const fullImagePath = path.join(labelPath, file);
        testingPaths.push({ path: fullImagePath, label });
      }
    }

    return { trainingPaths, testingPaths, allLabels };
  }

  /**
   * Generates a map from a given label to a unique integer
   * for use by the {@link labelToCategorical} function.
   *
   * Example:
   *
   * ```typescript
   * ["glioma", "meningioma", "notumor", "pituitary"]
   * ```
   * ==>
   * ```typescript
   * {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
   * ```
   *
   * @param allLabels A list of possible labels for the image
   *  data.
   * @returns A map from a given label to an integer.
   */
  private createLabelMap(allLabels: string[]): { [label: string]: number } {
    const labelsMap: { [label: string]: number } = {};
    for (let i = 0; i < allLabels.length; i++) {
      labelsMap[allLabels[i]] = i;
    }
    return labelsMap;
  }

  /**
   * "One-hot" encode a given label.
   *
   * Example:
   *
   * ```typescript
   * labelToCategorical("meningioma", {
   *   glioma: 0,
   *   meningioma: 1,
   *   notumor: 2,
   *   pituitary: 3,
   * });
   * ```
   * ==>
   * ```typescript
   * [0.0, 1.0, 0.0, 0.0]
   * ```
   *
   * @param label The label to convert to a categorical output.
   * @param labelsMap A map from a given label to an integer.
   * @returns A "one-hot" output representing a given image's
   *  desired output when run through the CNN.
   */
  private labelToCategorical(
    label: string,
    labelsMap: { [label: string]: number }
  ) {
    const toRet = [];
    for (const l in labelsMap) {
      if (l === label) {
        toRet.push(1.0);
      } else {
        toRet.push(0.0);
      }
    }
    return tf.tensor(toRet, undefined, "float32");
  }
}

/**
 * Read an image from the filesystem and turn it into a
 * {@link tf.Tensor}.
 *
 * @param path A path to an image to read.
 * @param channels The desired number of channels in the loaded
 *  image.
 * @returns A {@link tf.Tensor} representing the given image.
 */
export function readImage(path: string, channels?: number) {
  return tf.tidy(() => {
    const imageBuffer = fs.readFileSync(path);
    const tfimage = tf.node.decodeImage(imageBuffer, channels);

    return tfimage;
  });
}
