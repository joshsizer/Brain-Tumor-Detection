FROM nginx

ARG APP_ENV

# set up sites-available and sites enabled
RUN mkdir /etc/nginx/sites-available/
RUN mkdir /etc/nginx/sites-enabled/

# add each site-specific config to sites-available
RUN mv /etc/nginx/conf.d/default.conf /etc/nginx/sites-available/

# link each site in sites-available to sites-enabled
RUN ln -s /etc/nginx/sites-available/${APP_ENV}.conf /etc/nginx/sites-enabled/${APP_ENV}.conf
