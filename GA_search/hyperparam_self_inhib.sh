#!/bin/bash

arguments_path="arguments_self_inhib_agg.txt"

i=1  
while read ARGS; do
	# python LGMD_GA_search.py -f events_square_60_120_original_stimuli_frames_1000_aug.spikes $ARGS
	# python LGMD_GA_search.py -f events_square_15_120_original_stimuli_frames_1000_aug.spikes $ARGS
	# python LGMD_GA_search.py -f events_circle_19_120_original_stimuli_frames_1000_aug.spikes $ARGS
	# python LGMD_GA_search.py -f events_circle_15_120_original_stimuli_frames_1000_aug.spikes $ARGS
	# python LGMD_GA_search.py -f events_square_19_120_original_stimuli_frames_1000_aug.spikes $ARGS
	
	python LGMD_GA_search.py -f events_square_60_120_original_stimuli_frames_1000_aug.spikes $ARGS
	python LGMD_GA_search.py -f events_square_120_120_original_stimuli_frames_1000_aug.spikes $ARGS
	python LGMD_GA_search.py -f events_square_479_120_original_stimuli_frames_1000_aug.spikes $ARGS
	
	i=$((i+1))
done < $arguments_path