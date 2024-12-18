#!/bin/bash

# Name of the Python script to run
SCRIPT_NAME="./clustering.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME does not exist."
    exit 1
fi

# Run the specified Python script in the background with nohup and unbuffered output
if nohup python3 -u "$SCRIPT_NAME" > output.log 2>&1 & then
    # Print job status if the command was successful
    echo "Started $SCRIPT_NAME in the background. Logging to output.log"
else
    # Print an error message if the command failed
    echo "Failed to start $SCRIPT_NAME."
    exit 1
fi
