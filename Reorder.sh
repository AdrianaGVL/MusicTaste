###############################
# Author: Adriana Galán
# Music Taste Project
###############################
# This script is to move every seleted song into the desired folder to process it.

genre="pop"
new_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/used_dataset/$genre"
actual_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/mtg-jamendo-dataset/realdataset"
file="new_data/classify_$genre\_chosen.tsv"
echo "$file"
read -r || [[ -n $REPLY ]]
while IFS=$'\t' read -r track_id artist_id album_id path duration tags; do
    data_path="$actual_path/$path/$track_id"
    mv "$data_path" "$new_path"
done < "$file"