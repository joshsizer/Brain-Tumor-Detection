import React, { useEffect, useState } from "react";
import {
  ApolloClient,
  InMemoryCache,
  gql,
  from,
  HttpLink,
} from "@apollo/client/core";
import { onError } from "@apollo/client/link/error";
import ConfidenceBarChart from "../components/ConfidenceBarChart";

interface Props {}

const BASE_PATH = "/brain-tumor-detection";
const GRAPHQL_URL = `${BASE_PATH}/graphql`;
const EPOCH_VS_ACC_VS_LOSS = "epoch-loss-acc-graph.png";
const TEST_TRAIN_COUNT = "train-test-count-barchart.png";

const Dashboard: React.FC<Props> = () => {
  const [data, setData] = useState<any>();

  useEffect(() => {
    tryGetConfidences().then((data) => {
      setData(data);
    });
  }, []);

  const back = () => {
    window.location.href = `${BASE_PATH}`;
  };

  const tryGetConfidences = async () => {
    const data = (await getConfidence()).data.getConfidence as any[];
    const mapped = data.map((value) => {
      return {
        index: value.index,
        range: value.index * 0.025,
        count: value.count,
      };
    });
    console.log(mapped);
    mapped.sort(
      (
        a: { index: any; range: number; count: any },
        b: { index: any; range: number; count: any }
      ) => {
        return a.index > b.index ? 1 : a.index === b.index ? 0 : -1;
      }
    );
    return mapped;
  };

  return (
    <div>
      <button onClick={back}>Back</button>
      <h1>Dashboard</h1>
      <div style={{ width: 500, height: "auto" }}>
        <img
          src={`${BASE_PATH}/${EPOCH_VS_ACC_VS_LOSS}`}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
      <div style={{ paddingTop: 50, width: 500, height: "auto" }}>
        <img
          src={`${BASE_PATH}/${TEST_TRAIN_COUNT}`}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
      <div style={{ paddingTop: 50, width: 500, height: 400 }}>
        <p>Count of Predictions at Different Confidence Levels</p>
        <ConfidenceBarChart data={data} />
      </div>
    </div>
  );
};

export default Dashboard;

async function getConfidence() {
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
    uri: `${GRAPHQL_URL}`,
  });

  const client = new ApolloClient({
    link: from([errorLink, httpLink]),
    cache: new InMemoryCache(),
  });

  const getConfidenceQuery = gql`
    query getConfidence {
      getConfidence {
        index
        count
      }
    }
  `;

  const getConfidence = async () => {
    return client.query({
      query: getConfidenceQuery,
      errorPolicy: "all",
    });
  };

  return await getConfidence();
}
