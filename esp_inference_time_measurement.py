import os, time, serial, numpy as np, tensorflow as tf

MODEL_IO_FLOAT = False
USE_CONTEXT = True
USE_FOVEA = True
DATASET = "KTH"
EXPERIMENT = "both64_500_64_ctx64x64_fov32x32"
CONTEXT_SHAPE = (64, 64, 1)
FOVEA_SHAPE   = (32, 32, 1)
fold_index = 1

OUTPUT_DIR   = f"{DATASET}_results/{EXPERIMENT}"
TEST_DATA_DIR= os.path.join(OUTPUT_DIR, "test_data")
ESP_DIR      = os.path.join(OUTPUT_DIR, "esp32_ptq" if MODEL_IO_FLOAT else "esp32_qat")
os.makedirs(ESP_DIR, exist_ok=True)

MODEL_PATH = f"{DATASET}_results/{EXPERIMENT}/quantized_models_ptq/fold{fold_index}.tflite" if MODEL_IO_FLOAT else f"{DATASET}_results/{EXPERIMENT}/quantized_models_qat/fold_{fold_index}_qat_model.tflite"
NPZ_PATH   = os.path.join(TEST_DATA_DIR, f"fold_{fold_index}_test_data.npz")
ESP_PORT   = "/dev/ttyACM0"
BAUD_RATE  = 115200
BOOT_WAIT  = 2
CHUNK_SIZE = 256
TIME_TXT   = os.path.join(ESP_DIR, f"inference_time_fold{fold_index}.txt")

def get_io_qparams(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    ins = interp.get_input_details(); outs = interp.get_output_details()
    def qp(t):
        q = t.get("quantization_parameters", {})
        scale = float(q.get("scales", [1.0])[0]) if q.get("scales", []) else 1.0
        zp = int(q.get("zero_points", [0])[0]) if q.get("zero_points", []) else 0
        return {"scale": scale, "zero_point": zp, "dtype": t["dtype"]}
    return qp(ins[0]), qp(outs[0])

IN_Q, OUT_Q = get_io_qparams(MODEL_PATH)

def wait_for_tag(ser, tag, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = ser.readline()
        if not raw: continue
        try: line = raw.decode('utf-8', errors='ignore').strip()
        except: continue
        if tag in line: return line
    raise TimeoutError

def flatten_and_reshape(item, target_shape):
    if item is None: return np.empty((0,), dtype=np.float32)
    if isinstance(item, str):
        img = tf.io.read_file(item)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, target_shape[:2])
        img = tf.ensure_shape(img, target_shape)
        arr = img.numpy().astype(np.float32)
    else:
        arr = np.array(item, dtype=np.float32)
        if arr.ndim == 2: arr = arr[..., None]
        if arr.shape[:2] != target_shape[:2]: arr = tf.image.resize(arr, target_shape[:2]).numpy()
    return arr.reshape(-1).astype(np.float32)

def quantize_to_int8(x, scale, zp):
    q = np.round(x / scale + zp).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q

def load_dataset(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    y = d['y']; n = len(y)
    def gs(key_path, key_arr, enable):
        if not enable: return [None]*n
        if key_path in d: return d[key_path]
        if key_arr  in d: return d[key_arr]
        raise KeyError
    Xc = gs('X_context_paths','X_context',USE_CONTEXT)
    Xf = gs('X_fovea_paths','X_fovea',USE_FOVEA)
    m = min(len(Xc), len(Xf), len(y))
    return Xc[:m], Xf[:m]

def send_sample_for_time(ser, ctx_item, fov_item):
    ctx_f = flatten_and_reshape(ctx_item, CONTEXT_SHAPE) if USE_CONTEXT else np.empty(0, np.float32)
    fov_f = flatten_and_reshape(fov_item, FOVEA_SHAPE)   if USE_FOVEA   else np.empty(0, np.float32)
    if MODEL_IO_FLOAT:
        ctx_tx, fov_tx = ctx_f, fov_f
        as_int = False
    else:
        ctx_tx = quantize_to_int8(ctx_f, IN_Q["scale"], IN_Q["zero_point"])
        fov_tx = quantize_to_int8(fov_f, IN_Q["scale"], IN_Q["zero_point"]) if USE_FOVEA else np.empty(0, np.int8)
        as_int = True
    ser.write(f"S {len(ctx_tx)} {len(fov_tx)}\n".encode()); ser.flush()
    wait_for_tag(ser, "ACK S")
    def send_chunks(data, tag):
        for off in range(0, len(data), CHUNK_SIZE):
            chunk = data[off:off+CHUNK_SIZE]
            if as_int: csv = ",".join(str(int(v)) for v in chunk.astype(np.int16))
            else:      csv = ",".join(f"{float(v):.6f}" for v in chunk.astype(np.float32))
            ser.write(f"{tag} {off} {csv}\n".encode()); ser.flush()
            wait_for_tag(ser, f"ACK {tag}")
    if USE_CONTEXT: send_chunks(ctx_tx, "C")
    if USE_FOVEA:   send_chunks(fov_tx, "F")
    ser.write(b"E\n"); ser.flush()
    tline = wait_for_tag(ser, "RT:")
    t_us = int(tline.split("RT:",1)[1].strip())
    return t_us

if __name__ == '__main__':
    Xc, Xf = load_dataset(NPZ_PATH)
    idx = 0
    ser = serial.Serial(ESP_PORT, BAUD_RATE, timeout=0.2)
    time.sleep(BOOT_WAIT)
    ser.reset_input_buffer()
    t_us = send_sample_for_time(ser, Xc[idx] if USE_CONTEXT else None, Xf[idx] if USE_FOVEA else None)
    ser.close()
    with open(TIME_TXT, "w") as f:
        f.write(str(int(t_us)))
    print(t_us)
