# CD into the application folder and run a constant message script so that we don't get kicked off
# the job
cd /application/simpleml/ && (../build_scripts/constant_messaging.sh &)

# Run tests, and send results to coveralls.io
# COVERALLS_REPO_TOKEN=$COVERALLS_REPO_TOKEN TRAVIS_BRANCH=$TRAVIS_BRANCH BRANCH=$TRAVIS_BRANCH coveralls
cd /application/simpleml/ && nosetests --exe -c ../.noserc --with-coverage --cover-inclusive --cover-tests --cover-package=. unit_tests && COVERALLS_REPO_TOKEN=$1 TRAVIS_BRANCH=$2 BRANCH=$3 coveralls --rcfile=../.coveragerc
