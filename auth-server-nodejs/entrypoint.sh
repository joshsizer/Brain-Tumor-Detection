#! /bin/sh

# Following few lines from:
# https://gist.github.com/mihow/9c7f559807069a03e302605691f85572
# Loads in environment variables from a file.
if [ -f .env ]
then
  export $(cat .env | sed 's/#.*//g' | xargs)
fi

if [ ${APP_ENV} = ${APP_ENV_DEVELOPMENT} ]; then
    npm run watch &
    npm run dev
else
    npm run start
fi


