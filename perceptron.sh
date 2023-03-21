#!/bin/bash

arguments_path="arguments_self_inhib.txt"

i=1  
while read ARGS; do
	python3 LGMD.py -f events_circle_60_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python3 LGMD.py -f events_square_120_120_original_stimuli_fram_1000_augs.spikes $ARGS
	python3 LGMD.py -f events_square_15_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python3 LGMD.py -f events_square_479_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python3 LGMD.py -f events_square_60_120_original_stimuli_fram_1000_aug.spikes $ARGS
	
	python3 LGMD.py -f events_circle_60_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python3 LGMD.py -f events_square_60_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python3 LGMD.py -f events_square_479_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python3 LGMD.py -f events_square_120_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python3 LGMD.py -f events_square_15_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	
	i=$((i+1))
done < $arguments_path