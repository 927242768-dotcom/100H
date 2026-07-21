#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define BAR_WINDOW_BYTES (64 * 1024)
#define HEADER_OFFSET 0x000
#define FRAME_OFFSET 0x100
/*
 * 当前 FPGA 工程里 rx_top 的 ADDR_WIDTH 仍是 9，BAR0 读写地址以 16B 为单位，
 * 且前 16 个 word 预留给状态头，所以安全帧区大约是 (512 - 16) * 16 = 7936B。
 * 这里把 quickcheck 的默认分辨率限制在这个范围内，避免误把大图写穿。
 */
#define CURRENT_SAFE_FRAME_BYTES (7936u)

#define REG_CONTROL 0x000
#define REG_WIDTH 0x010
#define REG_HEIGHT 0x020
#define REG_THRESHOLD 0x030
#define REG_ROI_XY 0x040
#define REG_ROI_WH 0x050
#define REG_MORPH_CFG 0x060
#define REG_FRAME_BYTES 0x070

#define STATUS_SIGNATURE 0x54504650u
#define STATUS_BUSY (1u << 0)
#define STATUS_DONE (1u << 1)
#define STATUS_ERROR (1u << 2)
#define STATUS_CONTINUOUS (1u << 3)

typedef struct Options {
    const char *resource_root;
    uint64_t phys_addr;
    bool use_phys_addr;
    int width;
    int height;
    int threshold;
    double timeout_s;
    const char *output_dir;
} Options;

typedef struct StatusWords {
    uint32_t dwords[12];
} StatusWords;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s (--resource-root <sysfs-root> | --phys-addr <addr>) [--width N] [--height N] "
            "[--threshold N] [--timeout SEC] [--output-dir DIR]\n",
            prog);
}

static int parse_int_arg(const char *name, const char *value) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (value[0] == '\0' || end == NULL || *end != '\0') {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, value);
        exit(2);
    }
    return (int)parsed;
}

static uint64_t parse_u64_arg(const char *name, const char *value) {
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (value[0] == '\0' || end == NULL || *end != '\0') {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

static double parse_double_arg(const char *name, const char *value) {
    char *end = NULL;
    double parsed = strtod(value, &end);
    if (value[0] == '\0' || end == NULL || *end != '\0') {
        fprintf(stderr, "Invalid number for %s: %s\n", name, value);
        exit(2);
    }
    return parsed;
}

static Options parse_args(int argc, char **argv) {
    Options opt;
    int i;

    opt.resource_root = NULL;
    opt.phys_addr = 0;
    opt.use_phys_addr = false;
    opt.width = 64;
    opt.height = 64;
    opt.threshold = 96;
    opt.timeout_s = 0.5;
    opt.output_dir = "quickcheck_out";

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--resource-root") == 0 && i + 1 < argc) {
            opt.resource_root = argv[++i];
        } else if (strcmp(argv[i], "--phys-addr") == 0 && i + 1 < argc) {
            opt.phys_addr = parse_u64_arg("--phys-addr", argv[++i]);
            opt.use_phys_addr = true;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            opt.width = parse_int_arg("--width", argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            opt.height = parse_int_arg("--height", argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            opt.threshold = parse_int_arg("--threshold", argv[++i]);
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            opt.timeout_s = parse_double_arg("--timeout", argv[++i]);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            opt.output_dir = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(2);
        }
    }

    if (opt.resource_root == NULL && !opt.use_phys_addr) {
        print_usage(argv[0]);
        exit(2);
    }

    return opt;
}

static void join_path(char *dst, size_t dst_size, const char *dir, const char *leaf) {
    size_t dir_len = strlen(dir);
    int need_slash = (dir_len > 0 && dir[dir_len - 1] != '/');
    snprintf(dst, dst_size, "%s%s%s", dir, need_slash ? "/" : "", leaf);
}

static void *map_resource0(const char *resource_root, int *fd_out) {
    char resource0_path[1024];
    int fd;
    void *base;

    join_path(resource0_path, sizeof(resource0_path), resource_root, "resource0");
    fd = open(resource0_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", resource0_path, strerror(errno));
        return MAP_FAILED;
    }

    base = mmap(NULL, BAR_WINDOW_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "mmap %s failed: %s\n", resource0_path, strerror(errno));
        close(fd);
        return MAP_FAILED;
    }

    *fd_out = fd;
    return base;
}

static void *map_bar_from_physical(uint64_t phys_addr, int *fd_out) {
    int fd;
    void *base;

    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
        return MAP_FAILED;
    }

    base = mmap(NULL, BAR_WINDOW_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)phys_addr);
    if (base == MAP_FAILED) {
        fprintf(stderr, "mmap /dev/mem @0x%llx failed: %s\n",
                (unsigned long long)phys_addr, strerror(errno));
        close(fd);
        return MAP_FAILED;
    }

    *fd_out = fd;
    return base;
}

static void write32(volatile uint8_t *base, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(base + offset) = value;
    __sync_synchronize();
}

static uint32_t read32(volatile uint8_t *base, uint32_t offset) {
    return *(volatile uint32_t *)(base + offset);
}

static void write_bytes(volatile uint8_t *base, uint32_t offset, const uint8_t *src, size_t len) {
    memcpy((void *)(base + offset), src, len);
    msync((void *)base, BAR_WINDOW_BYTES, MS_SYNC);
    __sync_synchronize();
}

static void read_bytes(volatile uint8_t *base, uint32_t offset, uint8_t *dst, size_t len) {
    memcpy(dst, (const void *)(base + offset), len);
}

static void dump_status(volatile uint8_t *base, StatusWords *status) {
    int i;
    for (i = 0; i < 12; ++i) {
        status->dwords[i] = read32(base, HEADER_OFFSET + (uint32_t)(i * 4));
    }
}

static void print_status(const char *prefix, const StatusWords *status) {
    uint32_t status_bits = status->dwords[1];
    uint32_t width = status->dwords[2] & 0xFFFFu;
    uint32_t height = (status->dwords[2] >> 16) & 0xFFFFu;
    uint32_t frame_counter = status->dwords[3];
    uint32_t threshold = status->dwords[4] & 0xFFu;
    uint32_t morph_cfg = (status->dwords[4] >> 16) & 0xFFFFu;
    uint32_t frame_bytes = status->dwords[7];
    uint32_t active_pixels = status->dwords[8];

    printf(
        "%s: busy=%u done=%u error=%u continuous=%u width=%u height=%u "
        "frame_counter=%u threshold=%u morph_cfg=0x%04x frame_bytes=%u active_pixels=%u\n",
        prefix,
        (status_bits & STATUS_BUSY) ? 1u : 0u,
        (status_bits & STATUS_DONE) ? 1u : 0u,
        (status_bits & STATUS_ERROR) ? 1u : 0u,
        (status_bits & STATUS_CONTINUOUS) ? 1u : 0u,
        width,
        height,
        frame_counter,
        threshold,
        morph_cfg,
        frame_bytes,
        active_pixels);
}

static void build_test_frame(uint8_t *frame, int width, int height) {
    int x;
    int y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            int idx = y * width + x;
            int value = (x * 255) / ((width > 1) ? (width - 1) : 1);
            if (x >= width / 4 && x < width / 2 && y >= height / 4 && y < height / 2) {
                value = 220;
            }
            frame[idx] = (uint8_t)value;
        }
    }
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0777) == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "mkdir %s failed: %s\n", path, strerror(errno));
    return -1;
}

static int write_pgm(const char *path, int width, int height, const uint8_t *payload, size_t payload_size) {
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return -1;
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    if (fwrite(payload, 1, payload_size, fp) != payload_size) {
        fprintf(stderr, "write %s failed: %s\n", path, strerror(errno));
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1000000000.0);
}

static void sleep_ms(long ms) {
    struct timespec req;
    req.tv_sec = ms / 1000;
    req.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&req, NULL);
}

int main(int argc, char **argv) {
    Options opt = parse_args(argc, argv);
    int fd = -1;
    volatile uint8_t *bar0 = NULL;
    size_t frame_bytes;
    size_t padded_bytes;
    uint8_t *frame = NULL;
    uint8_t *padded = NULL;
    uint8_t *mask = NULL;
    StatusWords status;
    double deadline;
    size_t i;
    char input_path[1024];
    char mask_path[1024];
    int rc = 1;

    if (opt.width <= 0 || opt.height <= 0) {
        fprintf(stderr, "width/height must be positive\n");
        return 2;
    }

    frame_bytes = (size_t)opt.width * (size_t)opt.height;
    if (FRAME_OFFSET + frame_bytes > BAR_WINDOW_BYTES) {
        fprintf(stderr, "frame is too large for BAR0 window, please reduce width/height\n");
        return 2;
    }
    if (frame_bytes > CURRENT_SAFE_FRAME_BYTES) {
        fprintf(stderr,
                "frame is too large for current FPGA build: %zu bytes > %u bytes, "
                "please reduce width/height\n",
                frame_bytes, CURRENT_SAFE_FRAME_BYTES);
        return 2;
    }

    padded_bytes = frame_bytes;
    if ((padded_bytes % 16u) != 0u) {
        padded_bytes += 16u - (padded_bytes % 16u);
    }

    frame = (uint8_t *)malloc(frame_bytes);
    padded = (uint8_t *)calloc(1, padded_bytes);
    mask = (uint8_t *)malloc(frame_bytes);
    if (frame == NULL || padded == NULL || mask == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        goto cleanup;
    }

    if (opt.use_phys_addr) {
        printf("mapping BAR0 from /dev/mem at physical address 0x%llx\n",
               (unsigned long long)opt.phys_addr);
        bar0 = (volatile uint8_t *)map_bar_from_physical(opt.phys_addr, &fd);
    } else {
        printf("mapping BAR0 from sysfs resource: %s/resource0\n", opt.resource_root);
        bar0 = (volatile uint8_t *)map_resource0(opt.resource_root, &fd);
    }
    if (bar0 == MAP_FAILED) {
        bar0 = NULL;
        goto cleanup;
    }

    dump_status(bar0, &status);
    printf("header signature = 0x%08x\n", status.dwords[0]);
    if (status.dwords[0] != STATUS_SIGNATURE) {
        fprintf(stderr, "status signature mismatch, current sbit is not our custom FPGA logic\n");
        goto cleanup;
    }
    print_status("initial", &status);

    write32(bar0, REG_WIDTH, (uint32_t)opt.width);
    write32(bar0, REG_HEIGHT, (uint32_t)opt.height);
    write32(bar0, REG_THRESHOLD, (uint32_t)(opt.threshold & 0xFF));
    write32(bar0, REG_ROI_XY, 0);
    write32(bar0, REG_ROI_WH, 0);
    write32(bar0, REG_MORPH_CFG, 0);
    write32(bar0, REG_FRAME_BYTES, (uint32_t)frame_bytes);
    write32(bar0, REG_CONTROL, 1u << 2);

    build_test_frame(frame, opt.width, opt.height);
    memcpy(padded, frame, frame_bytes);
    write_bytes(bar0, FRAME_OFFSET, padded, padded_bytes);

    dump_status(bar0, &status);
    if (status.dwords[0] != STATUS_SIGNATURE) {
        fprintf(stderr, "status signature mismatch after frame upload\n");
        goto cleanup;
    }
    print_status("before_start", &status);

    write32(bar0, REG_CONTROL, 1u);

    deadline = now_seconds() + opt.timeout_s;
    while (now_seconds() < deadline) {
        dump_status(bar0, &status);
        if ((status.dwords[1] & STATUS_ERROR) != 0u) {
            print_status("error", &status);
            fprintf(stderr, "FPGA returned error status\n");
            goto cleanup;
        }
        if ((status.dwords[1] & STATUS_DONE) != 0u) {
            break;
        }
        sleep_ms(5);
    }

    if ((status.dwords[1] & STATUS_DONE) == 0u) {
        print_status("timeout", &status);
        fprintf(stderr, "wait FPGA done timeout\n");
        goto cleanup;
    }

    print_status("final", &status);
    read_bytes(bar0, FRAME_OFFSET, mask, frame_bytes);

    {
        size_t active_pixels = 0;
        for (i = 0; i < frame_bytes; ++i) {
            if (mask[i] != 0u) {
                ++active_pixels;
            }
        }
        printf("mask active pixels counted on host = %zu\n", active_pixels);
    }

    if (ensure_dir(opt.output_dir) != 0) {
        goto cleanup;
    }

    join_path(input_path, sizeof(input_path), opt.output_dir, "input.pgm");
    join_path(mask_path, sizeof(mask_path), opt.output_dir, "mask.pgm");

    if (write_pgm(input_path, opt.width, opt.height, frame, frame_bytes) != 0) {
        goto cleanup;
    }
    if (write_pgm(mask_path, opt.width, opt.height, mask, frame_bytes) != 0) {
        goto cleanup;
    }

    printf("input saved to %s\n", input_path);
    printf("mask saved to %s\n", mask_path);
    rc = 0;

cleanup:
    if (bar0 != NULL) {
        munmap((void *)bar0, BAR_WINDOW_BYTES);
    }
    if (fd >= 0) {
        close(fd);
    }
    free(frame);
    free(padded);
    free(mask);
    return rc;
}
