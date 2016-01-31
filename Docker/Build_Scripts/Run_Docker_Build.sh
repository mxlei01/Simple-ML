# CD into the application folder
cd /application/Application/ &&

# Run tests, and send results to coveralls.io
# COVERALLS_REPO_TOKEN=$COVERALLS_REPO_TOKEN TRAVIS_BRANCH=$TRAVIS_BRANCH BRANCH=$TRAVIS_BRANCH coveralls
coverage run --rcfile=/application/.coveragerc -m unit_tests.run_all_tests && 
COVERALLS_REPO_TOKEN=$1 TRAVIS_BRANCH=$2 BRANCH=$3 coveralls
