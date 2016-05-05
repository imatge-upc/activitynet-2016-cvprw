list=$1

youtube-dl -i -a $list \
-o '/imatge/amontes/work/datasets/ActivityNet/v1.3/videos/%(id)s.%(ext)s' \
-f bestvideo[ext=mp4] \
-u al.montes.gomez@gmail.com
