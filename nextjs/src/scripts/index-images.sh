#! /bin/bash

# ATTENTION!
# Please define the PGHOST, PGUSER, PGPASSWORD environment 
# variables before running this script, otherwise you won't
# be able to connect to the database.

# We need postgresql-client to command the database, and imagemagick
# to grab each image's dimensions.
apt-get -y update
apt-get -y install postgresql-client imagemagick

cd ./public

# This is painfully slow, because we are encurring the overhead
# of a database connection for each image. I should rather
# construct a list of values to use in the second half of the
# INSERT INTO command. I am, however, just learning bash, so
# this will do.
for CLASSIFICATION_PATH in ./images/*; do
    CLASSIFICATION=$(basename $CLASSIFICATION_PATH)
    for IMAGE in $CLASSIFICATION_PATH/*; do
        IMG_PATH=$(echo $IMAGE | sed 's/.//');
        IMG_SIZE="$(identify -ping -format '%w %h' $IMAGE)"
        read -a ARR <<< $IMG_SIZE
        WIDTH=${ARR[0]}
        HEIGHT=${ARR[1]}
        psql -c "INSERT INTO brain_tumor_image (path, classification, width, height) VALUES ('$IMG_PATH', '$CLASSIFICATION', $WIDTH, $HEIGHT);"
    done;
done;

psql -c "SELECT * FROM brain_tumor_image;"