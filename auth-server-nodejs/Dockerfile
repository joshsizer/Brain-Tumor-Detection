FROM node

ARG APP_ENV
ARG APP_ENV_DEVELOPMENT
ARG APP_ENV_PRODUCTION

ENV APP_ENV=$APP_ENV
ENV APP_ENV_DEVELOPMENT=$APP_ENV_DEVELOPMENT
ENV APP_ENV_PRODUCTION=$APP_ENV_PRODUCTION

EXPOSE 3000

WORKDIR /auth-server-nodejs

COPY ./ ./

RUN chmod u+x install.sh && \
    chmod u+x entrypoint.sh

RUN ./install.sh

# Beautiful explanation of entrypoint
# https://stackoverflow.com/a/21564990
ENTRYPOINT ["./entrypoint.sh"]
