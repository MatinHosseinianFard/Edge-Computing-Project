#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "fold1_model_data.h"

static constexpr bool kUseContext      = true;
static constexpr bool kUseFovea        = true;
static constexpr bool kModelIOIsFloat  = false;

static constexpr int  kTensorArenaSize = 512 * 1024;
static constexpr int  kMaxLineLen      = 8192;

namespace {
uint8_t* tensor_arena    = nullptr;
char*    line_buf        = nullptr;
const tflite::Model* model            = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_ctx_tensor = nullptr;
TfLiteTensor* input_fov_tensor = nullptr;
TfLiteTensor* output_tensor    = nullptr;
int ctx_expected  = 0;
int fov_expected  = 0;
int ctx_filled    = 0;
int fov_filled    = 0;
enum class RecvState { WAIT_HEADER, RECV_DATA };
RecvState state = RecvState::WAIT_HEADER;
}

static int ReadLineBlocking(char* buf, int max_len) {
    int idx = 0;
    while (idx < max_len - 1) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        buf[idx++] = static_cast<char>(c);
        if (c == '\n') break;
    }
    buf[idx] = '\0';
    return idx;
}

static int TensorElemCount(const TfLiteTensor* t) {
    if (!t || !t->dims) return 0;
    int count = 1;
    for (int i = 0; i < t->dims->size; ++i) count *= t->dims->data[i];
    return count;
}

static void SendAck(const char* tag, int value) {
    printf("ACK %s %d\n", tag, value);
    fflush(stdout);
}

static void SendError(const char* msg) {
    printf("ERR %s\n", msg);
    fflush(stdout);
}

static void TfLiteSetup() {
    tensor_arena = reinterpret_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    line_buf = reinterpret_cast<char*>(heap_caps_malloc(kMaxLineLen, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!tensor_arena || !line_buf) return;
    tflite::InitializeTarget();
    model = tflite::GetModel(fold1_model);
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddConcatenation();
    resolver.AddQuantize();
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return;
    if (kUseContext && kUseFovea) {
        input_fov_tensor = interpreter->input(0);
        input_ctx_tensor = interpreter->input(1);
    } else if (kUseContext) {
        input_ctx_tensor = interpreter->input(0);
    } else {
        input_fov_tensor = interpreter->input(0);
    }
    output_tensor = interpreter->output(0);
    printf("MODE:%s|IO:%s\n", kUseContext && kUseFovea ? "CTX+FOV" : (kUseContext ? "CTX" : "FOV"), kModelIOIsFloat ? "FLOAT" : "INT8");
    fflush(stdout);
}

static int ParseFloatsIntoFloat(float* dst, int offset, int max, const char* csv) {
    if (!dst || !csv || max <= 0 || offset < 0) return 0;
    int written = 0;
    char* tmp = strdup(csv);
    if (!tmp) return 0;
    char* saveptr = nullptr;
    for (char* tok = strtok_r(tmp, ",", &saveptr);
         tok && (offset + written) < max;
         tok = strtok_r(nullptr, ",", &saveptr)) {
        dst[offset + written++] = strtof(tok, nullptr);
    }
    free(tmp);
    return written;
}

static int ParseIntsIntoInt8(int8_t* dst, int offset, int max, const char* csv) {
    if (!dst || !csv || max <= 0 || offset < 0) return 0;
    int written = 0;
    char* tmp = strdup(csv);
    if (!tmp) return 0;
    char* saveptr = nullptr;
    for (char* tok = strtok_r(tmp, ",", &saveptr);
         tok && (offset + written) < max;
         tok = strtok_r(nullptr, ",", &saveptr)) {
        dst[offset + written++] = static_cast<int8_t>(atoi(tok));
    }
    free(tmp);
    return written;
}

static void ProtocolLoop() {
    int len = ReadLineBlocking(line_buf, kMaxLineLen);
    if (len <= 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    char* p = line_buf + strlen(line_buf) - 1;
    while (p >= line_buf && (*p == '\n' || *p == '\r')) *p-- = '\0';

    if (state == RecvState::WAIT_HEADER) {
        int cN = 0, fN = 0;
        if (sscanf(line_buf, "S %d %d", &cN, &fN) == 2) {
            int ce = kUseContext ? cN : 0;
            int fe = kUseFovea   ? fN : 0;
            if (kUseContext && input_ctx_tensor) ce = TensorElemCount(input_ctx_tensor);
            if (kUseFovea && input_fov_tensor)  fe = TensorElemCount(input_fov_tensor);
            ctx_expected = ce;
            fov_expected = fe;
            ctx_filled = 0;
            fov_filled = 0;
            state = RecvState::RECV_DATA;
            SendAck("S", 0);
        }
        return;
    }

    if (state == RecvState::RECV_DATA) {
        if (strcmp(line_buf, "E") == 0) {
            if (interpreter->Invoke() != kTfLiteOk) {
                SendError("Invoke");
            } else {
                printf("R:");
                int n = output_tensor->dims->data[output_tensor->dims->size - 1];
                if (kModelIOIsFloat) {
                    const float* pf = output_tensor->data.f;
                    for (int i = 0; i < n; ++i) printf(" %f", pf[i]);
                } else {
                    const int8_t* pi = output_tensor->data.int8;
                    for (int i = 0; i < n; ++i) printf(" %d", pi[i]);
                }
                printf("\n");
                fflush(stdout);
            }
            state = RecvState::WAIT_HEADER;
            return;
        }

        if (kUseContext && line_buf[0] == 'C') {
            int off;
            char* csv = strchr(strchr(line_buf, ' ') + 1, ' ') + 1;
            sscanf(line_buf, "C %d", &off);
            if (kModelIOIsFloat) {
                int wrote = ParseFloatsIntoFloat(input_ctx_tensor->data.f, off, ctx_expected, csv);
                ctx_filled = std::min(ctx_expected, off + wrote);
            } else {
                int wrote = ParseIntsIntoInt8(input_ctx_tensor->data.int8, off, ctx_expected, csv);
                ctx_filled = std::min(ctx_expected, off + wrote);
            }
            SendAck("C", ctx_filled);
            return;
        }

        if (kUseFovea && line_buf[0] == 'F') {
            int off;
            char* csv = strchr(strchr(line_buf, ' ') + 1, ' ') + 1;
            sscanf(line_buf, "F %d", &off);
            if (kModelIOIsFloat) {
                int wrote = ParseFloatsIntoFloat(input_fov_tensor->data.f, off, fov_expected, csv);
                fov_filled = std::min(fov_expected, off + wrote);
            } else {
                int wrote = ParseIntsIntoInt8(input_fov_tensor->data.int8, off, fov_expected, csv);
                fov_filled = std::min(fov_expected, off + wrote);
            }
            SendAck("F", fov_filled);
            return;
        }
    }
}

extern "C" void app_main() {
    TfLiteSetup();
    while (true) {
        ProtocolLoop();
    }
}
