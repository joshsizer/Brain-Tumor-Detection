import React, { useState, useEffect, useCallback, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  ApolloClient,
  InMemoryCache,
  gql,
  from,
  HttpLink,
} from "@apollo/client/core";
import { onError } from "@apollo/client/link/error";

interface Props {}

const HomePage: React.FC<Props> = () => {
  const [model, setModel] = useState<tf.LayersModel>();
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const img = useRef<HTMLImageElement>(null);
  const [imgURL, setImgURL] = useState<string>();
  const [imgLoaded, setImgLoaded] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<string>();
  const [actual, setActual] = useState<string>();

  const indexToLabel = useCallback((index: number) => {
    const map: { [index: number]: string } = {
      0: "glioma",
      1: "meningioma",
      2: "notumor",
      3: "pituitary",
    };
    return map[index];
  }, []);

  const doPrediction = useCallback(() => {
    if (img.current && imgLoaded && model && modelLoaded) {
      let imageTensor = tf.browser.fromPixels(img.current);
      imageTensor = tf.tidy(() => {
        imageTensor = tf.image.resizeBilinear(imageTensor, [128, 128]);
        imageTensor = tf.cast(imageTensor, "float32");
        imageTensor = imageTensor.div(tf.scalar(255));
        imageTensor = imageTensor.expandDims(0);
        return imageTensor;
      });

      const oneHotPred = (
        model.predict(imageTensor) as tf.Tensor<tf.Rank>
      ).dataSync();
      const maxIndex = tf.argMax(oneHotPred);
      setPrediction(indexToLabel(maxIndex.dataSync()[0]));
    }
  }, [img, imgLoaded, model, setPrediction, indexToLabel]);

  useEffect(() => {
    if (imgLoaded && modelLoaded) {
      doPrediction();
    }
  }, [imgLoaded, modelLoaded, doPrediction]);

  const loadModel = useCallback(() => {
    tf.loadLayersModel("http://localhost/model/model.json").then((newModel) => {
      setModelLoaded(true);
      setModel(newModel);
      console.log("Model loaded.");
    });
  }, [setModel, setModelLoaded]);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  const onImgLoaded = useCallback(() => {
    setImgLoaded(true);
    console.log("Image loaded.");
  }, [setImgLoaded]);

  // Uncomment if imgURL has a default value.
  //
  // Sometimes, the image element and its content are loaded
  // before the page's javascript. In this case, our component
  // must do the heavy lifting of checking if the image has
  // loaded yet, since the onLoad callback will not have been
  // registered yet.
  //
  // useEffect(() => {
  //   const image = img.current;
  //   if (!imgLoaded && image && image.complete) {
  //     onImgLoaded();
  //   }
  // }, [imgLoaded, img, onImgLoaded]);

  const changeImage = useCallback(async () => {
    setImgLoaded(false);
    const brainTumorImage = ((await getRandomImage()) as any).data
      .getRandomImage;
    const imgPath = brainTumorImage.path;
    const actualLabel = brainTumorImage.classification;

    setImgURL(`http://localhost${imgPath}`);
    setActual(actualLabel);
  }, [setImgLoaded, setImgURL]);

  useEffect(() => {
    changeImage();
  }, [changeImage]);

  return (
    <>
      <h1>Brain Tumor Detection</h1>
      <p>The model's name is: {model?.name}</p>
      <img src={imgURL} onLoad={onImgLoaded} ref={img} />
      <p>Prediction: {prediction}</p>
      <p>Actual: {actual}</p>
      <button onClick={changeImage}>Change Image!</button>
    </>
  );
};

export default HomePage;

async function getRandomImage() {
  const errorLink = onError(
    ({ graphQLErrors, networkError /**forward, operation**/ }) => {
      if (graphQLErrors)
        graphQLErrors.map(({ message, locations, path }) =>
          console.log(
            `[GraphQL error]: Message: ${message}, Location: ${locations}, Path: ${path}`
          )
        );

      if (networkError) console.log(`[Network error]: ${networkError}`);
      // forward(operation);
    }
  );

  const httpLink = new HttpLink({
    uri: `http://localhost/graphql`,
  });

  const client = new ApolloClient({
    link: from([errorLink, httpLink]),
    cache: new InMemoryCache(),
  });

  const getRandomImageQuery = gql`
    query getRandomImage {
      getRandomImage {
        id
        path
        classification
        width
        height
      }
    }
  `;

  const getRandomImage = async () => {
    return client.query({
      query: getRandomImageQuery,
      errorPolicy: "all",
    });
  };

  return await getRandomImage();
}
