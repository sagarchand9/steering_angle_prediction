#!/bin/bash

GPU_COUNT=8
THREAD=0

cd /data/notebook
for a in {0..1}
do
	for b in {0..2}
	do
		for c in {0..2}
		do
			for d in {0..2}
			do
				for e in {0..1}
				do
					for f in {0..1}
					do
						if ((THREAD == GPU_COUNT)); then
							THREAD=0
							wait
						fi
						./boom_goes_dynamite.py $THREAD $a $b $c $d $e $f &> "$a$b$c$d$e$f.txt" &
						echo "./boom_goes_dynamite.py $THREAD $a $b $c $d $e $f"
						THREAD=$((THREAD+1))
					done
				done
			done
		done
	done
done
wait
