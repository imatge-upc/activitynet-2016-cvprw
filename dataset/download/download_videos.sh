list=$1

srun youtube-dl -i -a $list \
-o '/imatge/amontes/work/datasets/ActivityNet/v1.3/videos/%(id)s.%(ext)s' \
-f bestvideo[ext=mp4] \
-u al.montes.gomez@gmail.com \
-p 321+creatividad!5115 > download.log 2>&1
