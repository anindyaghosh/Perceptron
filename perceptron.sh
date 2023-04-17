#!/bin/bash

arguments_path="arguments_self_inhib.txt"

i=1  
while read ARGS; do
	python LGMD.py -f events_circle_60_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python LGMD.py -f events_square_15_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python LGMD.py -f events_square_60_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python LGMD.py -f events_circle_19_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python LGMD.py -f events_circle_15_120_original_stimuli_fram_1000_aug.spikes $ARGS
	python LGMD.py -f events_square_19_120_original_stimuli_fram_1000_aug.spikes $ARGS
	
	
	python LGMD.py -f events_circle_60_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python LGMD.py -f events_square_60_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python LGMD.py -f events_square_15_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python LGMD.py -f events_circle_19_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python LGMD.py -f events_circle_15_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	python LGMD.py -f events_square_19_120_original_stimuli_fram_1000_aug_reverse.spikes $ARGS
	
	i=$((i+1))
done < $arguments_path
