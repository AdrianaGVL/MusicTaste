###############################
# Author: Adriana Galán
# Music Taste Project
###############################
# This script is to move every selected song into the desired folder to process it.

genre="Classical"
new_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/used_dataset/val/$genre"
actual_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/mtg-jamendo-dataset/realdataset"
file="/Users/agv/Documents/Estudios/Universidad/MLLB/${genre}_val.tsv"
echo "$file"
while IFS=$'\t' read -r track_id artist_id album_id path duration tags; do
    if [[ $path == "PATH" ]]; then
            continue
    fi
    filename="${path##*/}"
    mv "$actual_path/$path" "$new_path/$filename"
done < "$file"