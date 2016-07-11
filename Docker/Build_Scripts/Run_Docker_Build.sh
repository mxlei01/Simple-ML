# CD into the application folder and run a constant message script so that we don't get kicked off
# the job
cd /application/Application/ && (../Build_Scripts/Constant_Message.sh &)

# Run tests, and send results to coveralls.io
# COVERALLS_REPO_TOKEN=$COVERALLS_REPO_TOKEN TRAVIS_BRANCH=$TRAVIS_BRANCH BRANCH=$TRAVIS_BRANCH coveralls
coverage3 run --rcfile=/application/.coveragerc -m unit_tests.run_all_tests && 
COVERALLS_REPO_TOKEN=$1 TRAVIS_BRANCH=$2 BRANCH=$3 coveralls
