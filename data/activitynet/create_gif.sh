#!/bin/sh

palette="/tmp/palette.png"
filters="trim=start_frame=48:end_frame=64,scale=600:400:flags=lanczos"

ffmpeg -v warning -i $1 -vf "$filters,palettegen" -y $palette
ffmpeg -v warning -ss 3 -i $1 -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $2
