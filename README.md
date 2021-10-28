# Full-Stack-Template

Containerized starter code for rapid application development. Includes production and dev environments, as well as basic JWT authentication.

There's a heavy focus on utilizing Typescript wherever possible for ease of development.

Each service can run independently via containers or in a cluster with Docker-Compose.

## Usage

### Build

<pre><code>docker-compose build</code></pre>

or

<pre><code>docker-compose build --parallel</code></pre>

for a parallel build.

### Run

<pre><code>docker-compose up</code></pre>

### Environment

In the project root directory, you should place a .env file with the variables shown in example.env. For a development environment with hot-reloading and a GraphQL playground, set APP_ENV to 'development'. For a production environment, set APP_ENV to 'production'.

## Services

### NGINX

Acts as a router by parsing incoming URLs and marshaling requests to their respective services.

### PostgreSQL

The postgres database for long-term, structured data storage.

### GraphQL

Uses Apollo GraphQL to expose an API and TypeORM to bind to the PostgreSQL database. Two versions of this service are run in parallel: one which is always authenticated for use internally, and one which is reachable externally, via NGINX.

### Next.js

Starter Next.js project with a Login and Create Account page.

### Certbot

Automatically attempts to renew HTTPS certificates once they are initialized. In order to initialize, see "Setting up automatic TLS certificates"

## Sources

### Setting up automatic TLS certificates

https://pentacent.medium.com/nginx-and-lets-encrypt-with-docker-in-less-than-5-minutes-b4b8a60d3a71

## Security Considerations

I plan to work on these over the coming months.

### Access Token Secrets (auth-server-nodejs/src/tokenSecrets.ts)

This should be removed and the respective variables placed into a .env file, not tracked by git.

### Database Username and Password (postgres/database.env)

This should be changed to a example.env, left for the user to create their own .env file not tracked by git.

### User passwords for authentication

I am using bcrypt for hashing and salting passwords. Use at your own risk (and be really hesitant to use this system for storing OTHER PEOPLE's data!)

## Future considerations

I'd like to try out https://next-auth.js.org/ as the authentication mechanism.
