import * as tf from "@tensorflow/tfjs-node";

/**
 * Get a reference to a function for use in the processing
 * phase. The returned function is designed to be passed to
 * {@link tf.data.Dataset.map}.
 *
 * This specific implementation will resize an image using
 * {@link tf.image.resizeBilinear}.
 *
 * @param desiredImageShape The desired shape of the image
 *  which should match the CNN input layer's shape.
 * @returns A function that will resize a given image to the
 * desired shape.
 */
export default function imagePreprocessor(desiredImageShape: {
  height: number;
  width: number;
  channels: number;
}): (img: tf.TensorContainer) => tf.TensorContainer {
  return (img: tf.TensorContainer): tf.TensorContainer => {
    return tf.tidy(() => {
      img = tf.image.resizeBilinear(img as tf.Tensor<tf.Rank.R3>, [
        desiredImageShape.height,
        desiredImageShape.width,
      ]);
      img = tf.cast(img, "float32");
      img = img.div(tf.scalar(255));
      return img;
    });
  };
}
