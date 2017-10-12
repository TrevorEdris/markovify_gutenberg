#!/bin/bash

python markov.py -i input/war_of_the_worlds.txt -k 2 -s 4 -v
python markov.py -i input/war_of_the_worlds.txt -k 3 -s 4 -v
python markov.py -i input/war_of_the_worlds.txt -k 4 -s 4 -v

python markov.py -i input/moby_dick.txt -k 2 -s 4 -v
python markov.py -i input/moby_dick.txt -k 3 -s 4 -v
python markov.py -i input/moby_dick.txt -k 4 -s 4 -v
