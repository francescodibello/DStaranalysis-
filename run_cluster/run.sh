#!/bin/bash

# Number of iterations
iterations=50

# Command to execute
#command="condor_submit batch_pythia.sub"
command="condor_submit batch_pythia_unsmeared.sub"

# Loop to run the command 50 times
for ((i=1; i<=iterations; i++))
do
  echo "Running iteration $i..."
  $command
done

echo "Completed running the command $iterations times."

