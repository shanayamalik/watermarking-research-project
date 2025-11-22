Steps:
1. Download 5000 Coco2017 subset: https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing
And put ground truth and meta_data.json in a folder

2. Run scripts

Python generate_sd.py --caption_file /path/to/meta_data.json
python eval.py --mode fid --dir1 outputs/ --dir2 /dir/here ground_truth --output_file scores