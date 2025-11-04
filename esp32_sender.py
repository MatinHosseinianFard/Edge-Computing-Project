import os
import time
import pickle
import shutil
import serial
import numpy as np
from threading import Lock, Thread
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_IO_FLOAT = False
USE_CONTEXT = True
USE_FOVEA = True

USE_RANDOM_SUBSET = True
SUBSET_SIZE = 1000
np.random.seed(42)

DATASET = "UCF11"
EXPERIMENT = "reverseResolution64_500_64_ctx32x32_fov64x64"

CONTEXT_SHAPE = (32, 32, 1)
FOVEA_SHAPE   = (64, 64, 1)

fold_index = 1

CLASSES = [
    "basketball","biking","diving","golf_swing","horse_riding",
    "soccer_juggling","swing","tennis_swing","trampoline_jumping",
    "volleyball_spiking","walking"
] if DATASET == "UCF11" else ["boxing", "handclapping", "handwaving", "walking"]

OUTPUT_DIR = f"{DATASET}_results/{EXPERIMENT}"
TEST_DATA_DIR = os.path.join(OUTPUT_DIR, "test_data")
ESP_DIR = os.path.join(OUTPUT_DIR, "esp32_ptq" if MODEL_IO_FLOAT else "esp32_qat")
os.makedirs(ESP_DIR, exist_ok=True)

if MODEL_IO_FLOAT:
    MODEL_PATH = f"{DATASET}_results/{EXPERIMENT}/quantized_models_ptq/fold{fold_index}.tflite"
else:
    MODEL_PATH = f"{DATASET}_results/{EXPERIMENT}/quantized_models_qat/fold_{fold_index}_qat_model.tflite"

NPZ_PATH     = os.path.join(TEST_DATA_DIR, f"fold_{fold_index}_test_data.npz")
ESP_PORTS    = ["/dev/ttyACM0", "/dev/ttyACM1"]
BAUD_RATE    = 115200
BOOT_WAIT    = 2
CHUNK_SIZE   = 256
UPDATE_EVERY = 50

CM_FIG_PATH = os.path.join(ESP_DIR, f"esp32_fold_{fold_index}_confusion_matrix.png")
PROG_PATH   = os.path.join(ESP_DIR, "esp_progress.pkl")
BACKUP_PATH = os.path.join(ESP_DIR, "esp_progress_backup.pkl")

lock = Lock()

if os.path.exists(PROG_PATH):
    with open(PROG_PATH, 'rb') as f:
        prog = pickle.load(f)
    shared_true = prog.get('shared_true', [])
    shared_pred = prog.get('shared_pred', [])
    counts      = prog.get('counts', {port: 0 for port in ESP_PORTS})
else:
    shared_true = []
    shared_pred = []
    counts      = {port: 0 for port in ESP_PORTS}

def get_io_qparams(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    def qp(t):
        q = t.get("quantization_parameters", {})
        scale = float(q.get("scales", [1.0])[0]) if q.get("scales", []) else 1.0
        zp = int(q.get("zero_points", [0])[0]) if q.get("zero_points", []) else 0
        dtype = t["dtype"]
        return {"scale": scale, "zero_point": zp, "dtype": dtype}
    in0 = qp(ins[0])
    out0 = qp(outs[0])
    return in0, out0

IN_Q, OUT_Q = get_io_qparams(MODEL_PATH)

def wait_for_ack(ser, tag, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode('utf-8', errors='ignore').strip()
        except:
            continue
        if tag in line:
            return line
    raise TimeoutError

def flatten_and_reshape(item, target_shape):
    if item is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(item, str):
        img = tf.io.read_file(item)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, target_shape[:2])
        img = tf.ensure_shape(img, target_shape)
        arr = img.numpy().astype(np.float32)
    else:
        arr = np.array(item, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[:2] != target_shape[:2]:
            arr = tf.image.resize(arr, target_shape[:2]).numpy()
    return arr.reshape(-1).astype(np.float32)

def quantize_to_int8(x, scale, zp):
    q = np.round(x / scale + zp).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q

def dequant_from_int8(q, scale, zp):
    return (q.astype(np.int32) - int(zp)) * float(scale)

def send_sample(ser, ctx_item, fov_item):
    ctx_f = flatten_and_reshape(ctx_item, CONTEXT_SHAPE) if USE_CONTEXT else np.empty(0, np.float32)
    fov_f = flatten_and_reshape(fov_item,   FOVEA_SHAPE)   if USE_FOVEA   else np.empty(0, np.float32)
    if MODEL_IO_FLOAT:
        ctx_tx = ctx_f
        fov_tx = fov_f
    else:
        ctx_tx = quantize_to_int8(ctx_f, IN_Q["scale"], IN_Q["zero_point"])
        fov_tx = quantize_to_int8(fov_f, IN_Q["scale"], IN_Q["zero_point"]) if USE_FOVEA else np.empty(0, np.int8)
    ser.write(f"S {len(ctx_tx)} {len(fov_tx)}\n".encode()); ser.flush()
    wait_for_ack(ser, "ACK S")
    def send_chunks(data, tag, as_int):
        for offset in range(0, len(data), CHUNK_SIZE):
            chunk = data[offset:offset + CHUNK_SIZE]
            if as_int:
                csv = ",".join(str(int(x)) for x in chunk.astype(np.int16))
            else:
                csv = ",".join(f"{float(x):.6f}" for x in chunk.astype(np.float32))
            ser.write(f"{tag} {offset} {csv}\n".encode()); ser.flush()
            wait_for_ack(ser, f"ACK {tag}")
    if USE_CONTEXT:
        send_chunks(ctx_tx, 'C', as_int=not MODEL_IO_FLOAT)
    if USE_FOVEA:
        send_chunks(fov_tx, 'F', as_int=not MODEL_IO_FLOAT)
    ser.write(b"E\n"); ser.flush()
    line = wait_for_ack(ser, "R:")
    payload = line.split("R:", 1)[1].strip().split()
    if MODEL_IO_FLOAT:
        probs = np.array([float(p) for p in payload], dtype=np.float32)
    else:
        raw = np.array([int(p) for p in payload], dtype=np.int8)
        probs = dequant_from_int8(raw, OUT_Q["scale"], OUT_Q["zero_point"]).astype(np.float32)
    return int(np.argmax(probs))

def plot_confusion_matrix(true_labels, pred_labels, classes, save_path, fold):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(classes))))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Fold {fold} - Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    y    = data['y']
    length = len(y)
    def get_stream(path_key, array_key, enabled):
        if not enabled:
            return [None] * length
        if path_key in data:
            return data[path_key]
        if array_key in data:
            return data[array_key]
        raise KeyError
    X_ctx = get_stream('X_context_paths', 'X_context', USE_CONTEXT)
    X_fov = get_stream('X_fovea_paths',   'X_fovea',   USE_FOVEA)
    min_len = min(len(X_ctx), len(X_fov), len(y))
    return X_ctx[:min_len], X_fov[:min_len], y[:min_len]

def worker(port, indices, X_ctx, X_fov, y_true):
    ser = serial.Serial(port, BAUD_RATE, timeout=0.2)
    time.sleep(BOOT_WAIT)
    ser.reset_input_buffer()
    start_idx = counts.get(port, 0)
    remaining = indices[start_idx:]
    with tqdm(remaining, desc=f"Inference {port}", position=ESP_PORTS.index(port),
              initial=start_idx, total=len(indices), dynamic_ncols=True) as bar:
        for idx in bar:
            try:
                pred = send_sample(ser, X_ctx[idx], X_fov[idx])
            except Exception as e:
                bar.write(f"[{port}] Error at idx {idx}: {e}")
                time.sleep(0.2)
                continue
            with lock:
                shared_true.append(int(y_true[idx]))
                shared_pred.append(pred)
                counts[port] += 1
                with open(PROG_PATH, 'wb') as pf:
                    pickle.dump({
                        'shared_true': shared_true,
                        'shared_pred': shared_pred,
                        'counts': counts
                    }, pf)
                if counts[port] % UPDATE_EVERY == 0:
                    plot_confusion_matrix(shared_true, shared_pred, CLASSES, CM_FIG_PATH, fold_index)
                    shutil.copy(PROG_PATH, BACKUP_PATH)
                    acc = accuracy_score(shared_true, shared_pred)
                    pw, rw, fw, _ = precision_recall_fscore_support(shared_true, shared_pred, average='weighted', zero_division=0)
                    bar.write(f"Accuracy {acc:.4f}  Weighted P {pw:.4f} R {rw:.4f} F1 {fw:.4f}")
    ser.close()

if __name__ == '__main__':
    X_context, X_fovea, y_true = load_dataset(NPZ_PATH)
    total = len(X_context) if USE_CONTEXT else len(X_fovea)

    if USE_RANDOM_SUBSET:
        if SUBSET_SIZE > total:
            raise ValueError(f"SUBSET_SIZE ({SUBSET_SIZE}) cannot be greater than the total number of samples ({total}).")

        subset_indices = np.random.choice(total, SUBSET_SIZE, replace=False)
        split = SUBSET_SIZE // 2
        indices = {
            ESP_PORTS[0]: subset_indices[:split],
            ESP_PORTS[1]: subset_indices[split:]
        }
    else:
        split = total // 2
        indices = {
            ESP_PORTS[0]: list(range(0, split)),
            ESP_PORTS[1]: list(range(split, total))
        }

    threads = []
    for port in ESP_PORTS:
        t = Thread(target=worker, args=(port, indices[port], X_context, X_fovea, y_true), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    shutil.copy(PROG_PATH, os.path.join(ESP_DIR, f"esp_progress_final_{timestamp}.pkl"))
    
    plot_confusion_matrix(shared_true, shared_pred, CLASSES, CM_FIG_PATH, fold_index)
    acc    = accuracy_score(shared_true, shared_pred)
    report = classification_report(shared_true, shared_pred, target_names=CLASSES, zero_division=0, digits=4)
    print(f"Accuracy {acc:.4f}")
    print(report)
    
    output_content = f"Accuracy: {acc:.4f}\n\n"
    output_content += "Classification Report:\n"
    output_content += report
    
    txt_path = os.path.join(ESP_DIR, f"esp_progress_final_{timestamp}.pkl").replace(".pkl", ".txt")
    try:
        with open(txt_path, 'w') as f:
            f.write(output_content)
    except IOError as e:
        tqdm.write(f"Could not write report to {txt_path}: {e}")