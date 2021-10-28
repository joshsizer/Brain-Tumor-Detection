# Why GraphQL, as apposed to REST?

I chose a GraphQL API because of its flexibility and consistent typing.
Using typescript on the client, I can define default values and structures
for objects in the case the queries fail. In the cases that queries succeed,
the developer and the typescript engine know the exact shape which helps
create safer code. This allows for good documentation of expected behaivor,
as well as custom view bindings for easily implementing different views on
the same data.

On a related note, only one query needs to be executed as apposed to many for
a typical REST API.

https://www.apollographql.com/blog/graphql/basics/graphql-vs-rest/
