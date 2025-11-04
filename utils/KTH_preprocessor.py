import os
import cv2
from tqdm import tqdm
import re

DATASET_PATHS = [
    "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/boxing",
    "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/walking",
    "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/handclapping",
    "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/handwaving"
]
SEQUENCES_FILE = "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/00sequences.txt"

TARGET_SIZES = [(128, 128), (64, 64), (32, 32), (16, 16)]
CENTER_CROP_MAP = {(128, 128): (64, 64), (64, 64): (32, 32), (32, 32): (16, 16), (16, 16): (8, 8)}

TRAIN_SUBJECTS = {"person11", "person12", "person13", "person14", "person15", "person16", "person17", "person18"}
VAL_SUBJECTS = {"person19", "person20", "person21", "person23", "person24", "person25", "person01", "person04"}
TEST_SUBJECTS = {"person22", "person02", "person03", "person05", "person06", "person07", "person08", "person09", "person10"}

OUTPUT_DIR = "/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/KTH/preprocessed_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVITIES = {"boxing", "handclapping", "handwaving", "walking"}

total_frames_count = 0

def read_sequences_file(txt_path):
    sequences_dict = {}
    pattern = re.compile(r"(person\d+_([a-zA-Z]+)_d\d+)\s+frames\s+(.*)")
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.search(line)
            if match:
                video_name = match.group(1)
                action_name = match.group(2).lower()
                frames_str = match.group(3)
                if action_name not in ACTIVITIES:
                    continue
                frame_ranges = [tuple(map(int, seq.strip().split("-"))) for seq in frames_str.split(",")]
                sequences_dict[f"{video_name}_uncomp.avi"] = frame_ranges
    return sequences_dict

def get_data_split(subject_id):
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    elif subject_id in VAL_SUBJECTS:
        return "valid"
    elif subject_id in TEST_SUBJECTS:
        return "test"
    return None

def center_crop(img, crop_size):
    h, w = img.shape[:2]
    ch, cw = crop_size
    start_h, start_w = (h - ch) // 2, (w - cw) // 2
    return img[start_h:start_h + ch, start_w:start_w + cw]

def preprocess_and_save(video_path, frame_ranges, output_dir, action, subject_id):
    global total_frames_count
    split = get_data_split(subject_id)
    if split is None:
        return
    output_file = "base_name.txt"
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    with open(output_file, "a") as f:
        f.write(base_name + " " + f"{frame_ranges}" + "\n")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    all_starts = [rng[0] for rng in frame_ranges]
    all_ends = [rng[1] for rng in frame_ranges]
    global_start = min(all_starts)
    global_end = max(all_ends)
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_start - 1)
    for frame_num in range(global_start, global_end + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame {frame_num} from {video_path}")
            break
        total_frames_count += 1
        for tsize in TARGET_SIZES:
            resized_frame = cv2.resize(frame, tsize, interpolation=cv2.INTER_AREA)
            out_subdir = os.path.join(output_dir, split, f"{tsize[0]}x{tsize[1]}", action)
            os.makedirs(out_subdir, exist_ok=True)
            out_filename = f"{base_name}_frame_{frame_num:04d}.png"
            cv2.imwrite(os.path.join(out_subdir, out_filename), resized_frame)
            if tsize in CENTER_CROP_MAP:
                csize = CENTER_CROP_MAP[tsize]
                cropped_img = center_crop(resized_frame, csize)
                out_subdir_crop = os.path.join(output_dir, split, f"centerCrop_{csize[0]}x{csize[1]}", action)
                os.makedirs(out_subdir_crop, exist_ok=True)
                out_filename_crop = f"{base_name}_frame_{frame_num:04d}.png"
                cv2.imwrite(os.path.join(out_subdir_crop, out_filename_crop), cropped_img)
    cap.release()

def main():
    sequences_dict = read_sequences_file(SEQUENCES_FILE)
    for dataset_path in DATASET_PATHS:
        action = os.path.basename(dataset_path)
        for video_file in tqdm(os.listdir(dataset_path), desc=f"Processing {action}"):
            video_full_path = os.path.join(dataset_path, video_file)
            subject_id = video_file.split("_")[0]
            if not os.path.isfile(video_full_path):
                print(f"Video file not found: {video_full_path}")
                continue
            frame_ranges = sequences_dict.get(video_file, [])
            if frame_ranges:
                preprocess_and_save(video_full_path, frame_ranges, OUTPUT_DIR, action, subject_id)
    print(f"\nTotal frames processed: {total_frames_count}")

if __name__ == "__main__":
    main()
