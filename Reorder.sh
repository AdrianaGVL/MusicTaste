###############################
# Author: Adriana Galán
# Music Taste Project
###############################
# This script is to move every seleted song into the desired folder to process it.

genre="Techno"
new_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/used_dataset/$genre"
actual_path="/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/mtg-jamendo-dataset/realdataset"
file="new_data/${genre}_training.tsv"
echo "$file"
while IFS=$'\t' read -r track_id artist_id album_id path duration tags; do
    if [[ $path == "PATH" ]]; then
            continue
    fi
    filename="${path##*/}"
    mv "$actual_path/$path" "$new_path/$filename"
done < "$file"