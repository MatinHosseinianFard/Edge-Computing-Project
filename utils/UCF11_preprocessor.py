from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from tqdm import tqdm
import random

DATASET_ROOT = Path("/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/UCF11/UCF11_updated_mpg")
OUTPUT_DIR = Path("/mnt/60FE87C2FE878F4A/Uni/Master's/Term2/Edge/Replication/datasets/UCF11/preprocessed_frames2/train")
ACTIVITIES = ["basketball","biking","diving","golf_swing","horse_riding", 
              "soccer_juggling","swing","tennis_swing","trampoline_jumping", 
              "volleyball_spiking","walking"]
TARGET_SIZES = [(128, 128), (64, 64), (32, 32), (16, 16)]
CENTER_CROP_MAP = {(128, 128): (64, 64), (64, 64): (32, 32), (32, 32): (16, 16), (16, 16): (8, 8)}
SAVE_FORMAT = "image"
SHOW_PROGRESS = True

def centre_crop(img: np.ndarray, size: tuple):
    crop_w, crop_h = size
    h, w = img.shape[:2]
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img[top:top + crop_h, left:left + crop_w]

def save_frame(frame: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_FORMAT == "image":
        frame_uint8 = frame.astype(np.uint8)
        cv2.imwrite(str(path), frame_uint8)
    else:
        np.save(str(path), frame.astype(np.uint8, copy=False))

def discover_videos(dataset_root: Path, activity: str):
    activity_dir = dataset_root / activity
    for sub in activity_dir.iterdir():
        if not sub.is_dir() or sub.name.lower() == 'annotation':
            continue
        for ext in ('*.mp4', '*.mpg', '*.avi'):
            yield from sub.glob(ext)

def parse_annotation_frames(video_stem: str, activity: str):
    anno_file = DATASET_ROOT / activity / 'Annotation' / f"{video_stem}.xgtf"
    if not anno_file.exists():
        return set()
    ns = {'viper': 'http://lamp.cfar.umd.edu/viper#', 'data': 'http://lamp.cfar.umd.edu/viperdata#'}
    try:
        tree = ET.parse(str(anno_file)); root = tree.getroot()
    except ParseError:
        from lxml import etree
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        tree = etree.parse(str(anno_file), parser); root = tree.getroot()
    frames = set()
    for obj in root.findall('.//viper:object', ns):
        for attr in obj.findall('viper:attribute', ns):
            name = attr.get('name') or ''
            if not name.startswith(activity):
                continue
            for bv in attr.findall('data:bvalue', ns):
                if bv.get('value') != 'true':
                    continue
                span = bv.get('framespan')
                if not span:
                    continue
                start, end = map(int, span.split(':'))
                frames.update(range(start, end + 1))
    if not frames:
        total = None
        info_file = root.find('.//viper:sourcefile/viper:file[@name="Information"]', ns)
        if info_file is not None:
            num_attr = info_file.find('viper:attribute[@name="NUMFRAMES"]/data:dvalue', ns)
            if num_attr is not None and num_attr.get('value'):
                try:
                    total = int(num_attr.get('value'))
                except ValueError:
                    total = None
        if total:
            frames = set(range(1, total + 1))
    return frames

def process_video(video_path: Path, activity: str, valid_frames: set):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_idx = 0
    written = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total_frames, desc=video_path.stem, disable=not SHOW_PROGRESS)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx not in valid_frames:
            if SHOW_PROGRESS:
                pbar.update(1)
            continue
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for tsize in TARGET_SIZES:
            resized = cv2.resize(frame_gray, tsize, interpolation=cv2.INTER_AREA)
            ext = 'png' if SAVE_FORMAT == 'image' else 'npy'
            out_dir = OUTPUT_DIR / f"{tsize[0]}x{tsize[1]}" / activity
            save_frame(resized, out_dir / f"{video_path.stem}_frame_{frame_idx:04d}.{ext}")
            written += 1
            crop_sz = CENTER_CROP_MAP.get(tsize)
            if crop_sz:
                cropped = centre_crop(resized, crop_sz)
                crop_dir = OUTPUT_DIR / f"centerCrop_{crop_sz[0]}x{crop_sz[1]}" / activity
                save_frame(cropped, crop_dir / f"{video_path.stem}_frame_{frame_idx:04d}.{ext}")
                written += 1
        if SHOW_PROGRESS:
            pbar.update(1)
    if SHOW_PROGRESS:
        pbar.close()
    cap.release()
    return written

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for activity in ACTIVITIES:
        refs = []
        for video_path in discover_videos(DATASET_ROOT, activity):
            valid = parse_annotation_frames(video_path.stem, activity)
            for idx in valid:
                refs.append((video_path, idx))
        sample_count = len(refs) * 5 // 5
        random.seed(42)
        sampled = random.sample(refs, sample_count)
        refs_map = {}
        for vp, fi in sampled:
            refs_map.setdefault(vp, set()).add(fi)
        for video_path, frames in refs_map.items():
            total_written += process_video(video_path, activity, frames)
    print(f"\nPre-processing complete. Total frames saved: {total_written}")

if __name__ == "__main__":
    main()
