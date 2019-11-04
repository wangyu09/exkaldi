#!/bin/bash

inFile=$1
outFile=$2

tr [" "] ["\n"] < $inFile | tr ["+"] [" "] | awk '{print $1}' | tr ["\n"] [" "] | sed "s/  /\n/g" | sed "s/<sp> //g" > $outFile
