// Construct a schema, using GraphQL schema language
import { gql } from "apollo-server-express";

export const typeDefs = gql`
  type Query {
    getUser(filter: UserFilter): User
    verifyPassword(filter: UserFilter!, password: String!): Boolean
  }

  type Mutation {
    addUser(
      firstName: String
      lastName: String
      age: Int
      username: String!
      email: String!
      password: String!
    ): Boolean!
  }

  input UserFilter {
    ids: [ID]
    emails: [String]
  }

  type User {
    id: Int!
    firstName: String!
    lastName: String!
    age: Int!
    username: String!
    email: String!
    password: String!
    refreshCount: Int!
  }
`;
