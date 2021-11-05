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
  const [model, setModel] = useState<tf.LayersModel | undefined>(undefined);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [imgURL, setImgURL] = useState<string | undefined>(
    "http://localhost/images/glioma/Te-gl_0010.jpg"
  );
  const [imgLoaded, setImgLoaded] = useState<boolean>(false);
  const img = useRef<HTMLImageElement>(null);
  const [imgNum, setImgNum] = useState<number>(10);
  const [prediction, setPrediction] = useState<string | undefined>(undefined);

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
    console.log(`Image loaded? : ${imgLoaded}`);
    if (img.current && imgLoaded) {
      console.log("Doing prediction.");

      let imageTensor = tf.browser.fromPixels(img.current);
      imageTensor = tf.tidy(() => {
        imageTensor = tf.image.resizeBilinear(imageTensor, [128, 128]);
        imageTensor = tf.cast(imageTensor, "float32");
        imageTensor = imageTensor.div(tf.scalar(255));
        imageTensor = imageTensor.expandDims(0);
        return imageTensor;
      });

      if (model) {
        console.log("Model is not null!");
        const oneHotPred = (
          model.predict(imageTensor) as tf.Tensor<tf.Rank>
        ).dataSync();
        const maxIndex = tf.argMax(oneHotPred);
        setPrediction(indexToLabel(maxIndex.dataSync()[0]));
      }
    }
  }, [img, imgLoaded, model, setPrediction, indexToLabel]);

  const loadModel = useCallback(() => {
    if (!modelLoaded) {
      tf.loadLayersModel("http://localhost/model/model.json").then(
        (newModel) => {
          console.log("Model loaded.");
          setModelLoaded(true);
          setModel(newModel);
          doPrediction();
        }
      );
    }
  }, [modelLoaded, setModel, setModelLoaded, doPrediction]);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  const onImgLoaded = useCallback(() => {
    setImgLoaded(true);
    console.log("Image loaded.");
    doPrediction();
  }, [setImgLoaded, doPrediction]);

  // Sometimes, the image element and its content are loaded
  // before the page's javascript. In this case, our component
  // must do the heavy lifting of checking if the image has
  // loaded yet, since the onLoad callback will not have been
  // registered yet.
  useEffect(() => {
    const image = img.current;
    if (!imgLoaded && image && image.complete) {
      onImgLoaded();
    }
  }, [imgLoaded, img, onImgLoaded]);

  const changeImage = async () => {
    setImgLoaded(false);
    const imgPath = ((await getRandomImage()) as any).data.getRandomImage.path;
    console.log(imgPath);
    setImgURL(`http://localhost${imgPath}`);
  };

  return (
    <>
      <h1>Hello there!</h1>
      <p>The model's name is: {model?.name}</p>
      <img src={imgURL} onLoad={onImgLoaded} ref={img} />
      <p>Prediction: {prediction}</p>
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
