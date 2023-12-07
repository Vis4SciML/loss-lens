#!/bin/bash

mongorestore --username admin --password pass --authenticationDatabase admin --db losslensdb /docker-entrypoint-initdb.d/losslensdb/

