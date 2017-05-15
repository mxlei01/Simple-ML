[![Build Status](https://travis-ci.org/mxlei01/simpleml.svg?branch=master)](https://travis-ci.org/mxlei01/simpleml)
[![Code Health](https://landscape.io/github/mxlei01/simpleml/master/landscape.svg?style=flat)](https://landscape.io/github/mxlei01/simpleml/master)
[![Coverage Status](https://coveralls.io/repos/github/mxlei01/simpleml/badge.svg?branch=master)](https://coveralls.io/github/mxlei01/simpleml?branch=master)
[![Code Climate](https://codeclimate.com/github/mxlei01/simpleml/badges/gpa.svg)](https://codeclimate.com/github/mxlei01/simpleml)
[![Join the chat at https://gitter.im/mxlei01/simpleml](https://badges.gitter.im/mxlei01/simpleml.svg)](https://gitter.im/mxlei01/simpleml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Simple-ML

A simple machine learning library written in Python

# To run unit tests

nosetests -c ../.noserc --with-coverage --cover-inclusive --cover-tests --cover-package=. --process-timeout 600000 --processes 32 unit_tests

# To Run a single unit test

nosetests -c ../.noserc -s --with-coverage --cover-inclusive --cover-tests --cover-package=. --process-timeout 600000 --processes 32 unit_tests.clustering.nearest_neighbor

