#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
python receiverPositioning.py 1 rec1.png
python receiverPositioning.py 2 rec2.png
python receiverPositioning.py 3 rec3.png
python receiverPositioning.py 5 rec5.png
python receiverPositioning.py 7 rec7.png
python receiverPositioning.py 9 rec9.png
python receiverPositioning.py 15 rec15.png
python receiverPositioning.py 20 rec20.png
python receiverPositioning.py 30 rec30.png

convert +append rec1.png rec2.png rec3.png out1.png
convert +append rec5.png rec7.png rec9.png out2.png
convert +append rec15.png rec20.png rec30.png out3.png
convert -append out1.png out2.png out3.png AllCombinedRecPos.png
rm out1.png out2.png out3.png
