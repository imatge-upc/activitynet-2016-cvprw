USER=$1
PASSWORD=$2

OUTPUT_DIR=${3:-'../data/activitynet'}

mkdir -p OUTPUT_DIR

youtube-dl -i -a videos_ids.lst \
-o $OUTPUT_DIR'/%(id)s.%(ext)s' \
-f bestvideo[ext=mp4] \
-u $USER \
-p $PASSWORD
