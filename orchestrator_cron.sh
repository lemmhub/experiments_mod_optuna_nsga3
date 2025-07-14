#!/bin/bash

# Run orchestrator every 10 seconds
for i in {1..6}; do
    /path/to/orchestrator.sh
    sleep 10
done
