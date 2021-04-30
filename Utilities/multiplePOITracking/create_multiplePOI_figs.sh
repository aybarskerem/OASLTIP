#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
python by_multiplePOI.py
convert +append Person1.png Person2.png out1.png
convert +append Person3.png Person4.png out2.png
convert +append Person5.png Person6.png out3.png
convert -append out1.png out2.png out3.png AllPeopleCombined.png