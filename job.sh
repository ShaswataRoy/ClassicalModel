#!/usr/bin/bash

cat './pbd_files.csv'|parallel --jobs 256  --joblog chimera_${timestamp}.out bash chimera_step.sh {}