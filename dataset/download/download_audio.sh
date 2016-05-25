list=$1

srun youtube-dl -i -a $list \
-o '/imatge/amontes/work/datasets/ActivityNet/v1.3/audio/%(id)s.%(ext)s' \
--extract-audio --audio-format wav \
-u al.montes.gomez@gmail.com \
-p 321+creatividad!5115 > download_2.log 2>&1
