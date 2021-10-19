import * as tf from "@tensorflow/tfjs-node";

/**
 * Define a convolutional neural network by extending a
 * {@link tf.Sequential} model. Observe the constructor for
 * the network's architecture.
 *
 * Great thanks to
 * https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html#0
 * for a great CNN example which I used as a starting point
 * for this model.
 */
export default class BrainTumorModel extends tf.Sequential {
  constructor(inputImageShape: {
    height: number;
    width: number;
    channels: number;
  }) {
    super();

    const { height, width, channels } = inputImageShape;

    super.add(
      tf.layers.conv2d({
        inputShape: [width, height, channels],
        kernelSize: 5,
        filters: 4,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 4;
    super.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      })
    );

    const optimizer = tf.train.adam();
    super.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }
}
