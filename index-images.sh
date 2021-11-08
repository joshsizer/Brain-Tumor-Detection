#! /bin/sh

should_exit=false

# How to test for unset variable:
# https://stackoverflow.com/a/13864829
if [ -z ${PGHOST+x} ]; 
then 
    echo "ERROR: PGHOST is not set."; 
    should_exit=true;
fi;

if [ -z ${PGDATABASE+x} ]; 
then 
    echo "ERROR: PGDATABASE is not set."; 
    should_exit=true;
fi;

if [ -z ${PGUSER+x} ]; 
then 
    echo "ERROR: PGUSER is not set."; 
    should_exit=true;
fi;

if [ -z ${PGPASSWORD+x} ]; 
then 
    echo "ERROR: PGPASSWORD is not set."; 
    should_exit=true;
fi;

if [ "$should_exit" = true ];
then
    exit 1;
fi;

docker exec -it \
    -e PGHOST=$PGHOST \
    -e PGDATABASE=$PGDATABASE \
    -e PGUSER=$PGUSER \
    -e PGPASSWORD=$PGPASSWORD \
    btd-nextjs \
    bash ./src/scripts/index-images.sh
