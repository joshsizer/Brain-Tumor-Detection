import express, { request } from "express";
import cookieParser from "cookie-parser";
import { sign, verify } from "jsonwebtoken";
import { ACCESS_TOKEN_SECRET, REFRESH_TOKEN_SECRET } from "./tokenSecrets";

// apollo-client is expecting a browser
import "cross-fetch/polyfill";
import {
  ApolloClient,
  InMemoryCache,
  gql,
  from,
  HttpLink,
} from "@apollo/client/core";
import { onError } from "@apollo/client/link/error";

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

const GRAPHQL_HOSTNAME = process.env.GRAPHQL_INTERNAL_HOSTNAME;
const GRAPHQL_PORT = process.env.GRAPHQL_INTERNAL_PORT;
const GRAPHQL_QUERY_PATH = process.env.GRAPHQL_QUERY_PATH;

const httpLink = new HttpLink({
  uri: `http://${GRAPHQL_HOSTNAME}:${GRAPHQL_PORT}/${GRAPHQL_QUERY_PATH}`,
});

const client = new ApolloClient({
  link: from([errorLink, httpLink]),
  cache: new InMemoryCache(),
});

/*
 * username: make
 * password: model
 * user: hotsauce
 * database: cupboard
 * authentication: conviction
 * access:
 * refresh:
 * token: car keys
 */
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

app.post("/login", async (req, res) => {
  const email = req.body.email;
  const password = req.body.password;

  console.log(`Email: ${email}, password: ${password}`);
  const verifyPasswordQuery = gql`
    query verifyPassword($email: String, $password: String!) {
      verifyPassword(filter: { emails: [$email] }, password: $password)
    }
  `;

  const verifyPassword = async () => {
    return client.query({
      query: verifyPasswordQuery,
      variables: {
        email: email,
        password: password,
      },
      errorPolicy: "all",
    });
  };

  const passwordOk = (await verifyPassword()).data.verifyPassword;
  console.log("Password OK?: %j", passwordOk);

  if (!passwordOk) {
    res.status(401);
    res.send({
      success: false,
      status:
        "You failed to convince me that you're the owner of this hotsauce",
      message: "Please provide a correct make and model.",
      error: "You don't know what you're talking about.",
      data: {},
    });
    return;
  }

  const getUserQuery = gql`
    query getUser($email: String) {
      getUser(filter: { emails: [$email] }) {
        id
        refreshCount
      }
    }
  `;

  const getUser = async () => {
    return client.query({
      query: getUserQuery,
      variables: {
        email: email,
      },
      errorPolicy: "all",
    });
  };

  const result = await getUser();

  if (!result.data.getUser) {
    res.status(401);
    res.send({
      success: false,
      status:
        "You failed to convince me that you're the owner of this hotsauce",
      message: "Server-ish error.",
      error: "Could not retrieve hotsauce from our cupboard.",
      data: {},
    });
    return;
  }

  const accessToken = sign(
    { userId: result.data.getUser.id },
    ACCESS_TOKEN_SECRET,
    {
      expiresIn: "15m",
    }
  );
  const refreshToken = sign(
    {
      userId: result.data.getUser.id,
      refreshCount: result.data.getUser.refreshCount,
    },
    REFRESH_TOKEN_SECRET,
    { expiresIn: "4h" }
  );

  res.cookie("access-token", accessToken, { maxAge: 60 * 15 * 1000 });
  res.cookie("refresh-token", refreshToken, { maxAge: 60 * 60 * 4 * 1000 });

  res.status(200);
  res.send({
    success: true,
    status: "Tasty sauce, sir.",
    message: "We found the condiment!",
    error: undefined,
    data: {},
  });
});

app.get("/userinfo", async (req, res) => {
  const accessToken = req.cookies["access-token"];
  const refreshToken = req.cookies["refresh-token"];

  if (!accessToken && !refreshToken) {
    res.status(401);
    res.send({
      success: false,
      status: "That request looks a little bland. May I suggest some hotsauce?",
      message: "Come back with something tastier.",
      error: "The request lacked adequate spice.",
      data: {},
    });
    return;
  }

  let accessTokenData: any;
  try {
    accessTokenData = verify(accessToken, ACCESS_TOKEN_SECRET) as {
      userId: Number;
    };
  } catch (error) {
    console.log("Access token error: " + error);
  }

  if (accessTokenData) {
    console.log(accessTokenData);
    res.status(200);
    res.send({
      success: true,
      status: "Authorization succeeded.",
      message: "Access-token valid.",
      error: undefined,
      data: { user: { id: accessTokenData.userId } },
    });
    return;
  }

  let refreshTokenData: any;

  try {
    refreshTokenData = verify(refreshToken, REFRESH_TOKEN_SECRET) as {
      userId: Number;
      refreshCount: Number;
    };
  } catch (error) {
    console.log("Refresh token error: " + error);
  }

  if (refreshTokenData) {
    console.log("Refresh token data: %j", refreshTokenData);
    const query = gql`
      query getUser($id: ID) {
        getUser(filter: { ids: [$id] }) {
          refreshCount
        }
      }
    `;

    const request = async () => {
      return client.query({
        query: query,
        variables: {
          id: refreshTokenData.userId,
        },
        errorPolicy: "all",
      });
    };

    const response = await request();

    if (response.data.getUser.refreshCount === refreshTokenData.refreshCount) {
      res.status(200);
      res.send({
        success: true,
        status: "Authorization succeeded.",
        message: "Refresh-token valid.",
        error: undefined,
        data: { user: { id: refreshTokenData.userId } },
      });
      return;
    }
  }

  res.status(401);
  res.send({
    success: false,
    status: "Authorization failed",
    message: "Please log in again.",
    error: "Invalid access and refresh-token.",
    data: {},
  });
});

app.post("/add-user", async (req, res) => {
  const username = req.body.username;
  const email = req.body.email;
  const password = req.body.password;

  const addUserMutation = gql`
    mutation addUser($username: String!, $email: String!, $password: String!) {
      addUser(username: $username, email: $email, password: $password)
    }
  `;

  const addUser = async () => {
    return client.mutate({
      mutation: addUserMutation,
      variables: {
        username: username,
        email: email,
        password: password,
      },
      errorPolicy: "all",
    });
  };

  const addUserRet = await addUser();
  console.log("Add user return value: ");
  console.log(addUserRet);
});

if (process.env.SERVER_PORT === undefined) {
  throw new Error("Please specify SERVER_PORT environment variable.");
}

if (process.env.SERVER_HOST === undefined) {
  throw new Error("Please specify SERVER_HOST environment variable.");
}

const PORT = parseInt(process.env.SERVER_PORT);
const HOST = process.env.SERVER_HOST;

app.listen(PORT, HOST, () => {
  console.log(`Auth server listening at http://${HOST}:${PORT}`);
});
