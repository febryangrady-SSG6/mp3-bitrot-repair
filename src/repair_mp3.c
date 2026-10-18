/*
 * repair_mp3.c — Automated 4-Pass MP3 Bitstream, Side-Info & Acoustic De-Clicking Repair Engine
 *
 * Repairs SD card / NAND flash bit-rot in MP3 files across four passes:
 *   - Pre-Pass: Validates embedded ID3v2 APIC PNG chunk bounds and IEEE 802.3 CRC-32 checksums,
 *               neutralizing broken APIC headers ("APIC" -> "TXXX") so decoders do not fail.
 *               Also recovers zeroed-out Frame 0 headers right after ID3v2.
 *   - Pass 1:   Reconstructs corrupted 4-byte MPEG frame headers and exact frame boundaries
 *               using Viterbi dynamic programming over padding bits (0/1) for CBR streams,
 *               joint current+next header Hamming scoring for VBR streams, and clean EOF
 *               stub/ID3v1 TAG handling.
 *   - Pass 2a:  Repairs Layer III side-information and bit-reservoir invariants:
 *               * main_data_begin <= reservoir_bytes (clearing flipped high bits)
 *               * big_values <= 288
 *               * Illegal window_switching_flag=1 & block_type=0
 *               * Reserved Huffman table_select (4 -> 0, 14 -> 15)
 *               * sum(part2_3_length) <= available_bits (prioritizing corrupted silent granules)
 *   - Pass 2b:  40-frame sliding-window acoustic de-clicking pass:
 *               * Resets bit-flipped silent granules / Mid-Side silent channels (global_gain=210
 *                 or part2_3_length=0 with big_values>0) and restores stolen bits to ch=0
 *               * Clamps local global_gain and big_values spikes relative to 40-frame window
 *               * Repairs high-linbits Huffman table_select bit-flips (e.g. 26/27 -> 30/31)
 *                 and false window_switching_flag=1 bit-flips on long blocks
 *   - Pass 3:   Recomputes and updates the embedded LAME / Lavc Info/Xing Audio CRC-16
 *               (AV_CRC_16_ANSI_LE) while preserving file ownership (uid:gid) and timestamps.
 *
 * Build:
 *   gcc -O3 -Wall -Wextra -fopenmp src/repair_mp3.c -o repair_mp3 -lavutil
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <libavutil/crc.h>
#include <omp.h>

static const int br_tab[3][16] = {
    {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0},
    {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
    {0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0}
};

static const int sr_tab[4][4] = {
    {11025, 12000, 8000, 0},
    {0, 0, 0, 0},
    {22050, 24000, 16000, 0},
    {44100, 48000, 32000, 0}
};

static uint32_t get_bits(const uint8_t *buf, int bit_pos, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++) {
        int byte_idx = (bit_pos + i) >> 3;
        int bit_idx = 7 - ((bit_pos + i) & 7);
        val = (val << 1) | ((buf[byte_idx] >> bit_idx) & 1);
    }
    return val;
}

static void set_bits(uint8_t *buf, int bit_pos, int n, uint32_t val) {
    for (int i = 0; i < n; i++) {
        int byte_idx = (bit_pos + i) >> 3;
        int bit_idx = 7 - ((bit_pos + i) & 7);
        uint32_t bit = (val >> (n - 1 - i)) & 1;
        buf[byte_idx] = (buf[byte_idx] & ~(1 << bit_idx)) | (bit << bit_idx);
    }
}

static int parse_header(const uint8_t *h, int *ver, int *layer, int *prot, int *bi_out, int *si_out, int *pad_out, int *cm_out, int *fsize) {
    if (h[0] != 0xFF || (h[1] & 0xE0) != 0xE0) return 0;
    int v = (h[1] >> 3) & 3;
    int l = (h[1] >> 1) & 3;
    int p = h[1] & 1;
    int bi = (h[2] >> 4) & 0xF;
    int si = (h[2] >> 2) & 3;
    int pad = (h[2] >> 1) & 1;
    int emp = h[3] & 3;
    if (v == 1 || l == 0 || bi == 0 || bi == 15 || si == 3 || emp == 2) return 0;
    int srate = sr_tab[v][si];
    if (!srate) return 0;
    int brate = 0;
    if (l == 1) {
        brate = (v == 3) ? br_tab[0][bi] : br_tab[1][bi];
        *fsize = ((v == 3 ? 144000 : 72000) * brate) / srate + pad;
    } else if (l == 2) {
        brate = (v == 3) ? br_tab[2][bi] : br_tab[1][bi];
        *fsize = (144000 * brate) / srate + pad;
    } else {
        return 0;
    }
    *ver = v; *layer = l; *prot = p; *bi_out = bi; *si_out = si; *pad_out = pad;
    *cm_out = (h[3] >> 6) & 3;
    return 1;
}

static int calc_fsize(int v, int l, int bi, int si, int pad) {
    int srate = sr_tab[v][si];
    int brate = (l == 1) ? ((v == 3) ? br_tab[0][bi] : br_tab[1][bi]) : ((v == 3) ? br_tab[2][bi] : br_tab[1][bi]);
    if (l == 1) return ((v == 3 ? 144000 : 72000) * brate) / srate + pad;
    return (144000 * brate) / srate + pad;
}

static int popcnt32(uint32_t x) {
    return __builtin_popcount(x);
}

static uint32_t make_hdr_u32(int v, int l, int p, int bi, int si, int pad, int cm, uint8_t b3_low6) {
    uint8_t h[4];
    h[0] = 0xFF;
    h[1] = 0xE0 | ((v & 3) << 3) | ((l & 3) << 1) | (p & 1);
    h[2] = ((bi & 0xF) << 4) | ((si & 3) << 2) | ((pad & 1) << 1);
    h[3] = ((cm & 3) << 6) | (b3_low6 & 0x3F);
    return ((uint32_t)h[0] << 24) | ((uint32_t)h[1] << 16) | ((uint32_t)h[2] << 8) | h[3];
}

static uint32_t read_u32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static void write_u32(uint8_t *p, uint32_t val) {
    p[0] = (val >> 24) & 0xFF;
    p[1] = (val >> 16) & 0xFF;
    p[2] = (val >> 8) & 0xFF;
    p[3] = val & 0xFF;
}

static uint32_t png_crc32(const uint8_t *buf, size_t len) {
    uint32_t c = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; i++) {
        c ^= buf[i];
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
    }
    return c ^ 0xFFFFFFFFU;
}

// Check if at pos there is a chain of >= 3 valid frames with (v0, l0, si0)
static int is_valid_chain(const uint8_t *data, long end_pos, long pos, int v0, int l0, int si0) {
    long cur = pos;
    for (int k = 0; k < 3; k++) {
        if (cur == end_pos) return 1;
        if (cur + 4 > end_pos) return (k >= 1);
        if (memcmp(data + cur, "TAG", 3) == 0 || memcmp(data + cur, "APET", 4) == 0 ||
            memcmp(data + cur, "ID3", 3) == 0 || memcmp(data + cur, "LYRI", 4) == 0)
            return 1;
        int v, l, p, bi, si, pad, cm, fs;
        if (!parse_header(data + cur, &v, &l, &p, &bi, &si, &pad, &cm, &fs)) return 0;
        if (v != v0 || l != l0 || si != si0) return 0;
        cur += fs;
    }
    return 1;
}

/* Pass 3 (rewriting the embedded LAME/Lavc audio CRC) is opt-in: the original CRC is the only
 * ground-truth oracle in the file, so by default we preserve it instead of forging a passing value. */
static int g_update_crc = 0;

int repair_file(const char *in_path, const char *out_path) {
    struct stat st;
    int has_stat = (stat(in_path, &st) == 0);

    FILE *f = fopen(in_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return -1; }
    uint8_t *data = malloc(fsize + 64);
    if (!data || fread(data, 1, fsize, f) != (size_t)fsize) {
        fclose(f); free(data); return -1;
    }
    fclose(f);
    memset(data + fsize, 0, 64);

    long pos = 0;
    while (pos + 10 <= fsize && memcmp(data + pos, "ID3", 3) == 0) {
        long sz = ((data[pos+6] & 0x7f) << 21) | ((data[pos+7] & 0x7f) << 14) |
                  ((data[pos+8] & 0x7f) << 7)  | (data[pos+9] & 0x7f);
        for (long i = pos + 10; i + 16 < pos + 10 + sz && i + 16 < fsize; i++) {
            if (memcmp(data + i, "\x89PNG\r\n\x1a\n", 8) == 0) {
                long png_p = i + 8;
                int png_bad = 1;
                while (png_p + 12 <= pos + 10 + sz && png_p + 12 <= fsize) {
                    uint32_t clen = read_u32(data + png_p);
                    if (clen > (uint32_t)(pos + 10 + sz - png_p - 12)) { png_bad = 1; break; }
                    uint32_t exp_crc = read_u32(data + png_p + 8 + clen);
                    uint32_t calc_crc = png_crc32(data + png_p + 4, 4 + clen);
                    if (exp_crc != calc_crc) { png_bad = 1; break; }
                    if (memcmp(data + png_p + 4, "IEND", 4) == 0) { png_bad = 0; break; }
                    png_p += 12 + clen;
                }
                if (png_bad && i >= pos + 14) {
                    for (long k = i - 1; k >= pos + 10 && k >= i - 512; k--) {
                        if (memcmp(data + k, "APIC", 4) == 0) {
                            memcpy(data + k, "TXXX", 4);
                            break;
                        }
                    }
                }
            }
        }
        pos += 10 + sz;
    }

    long end_pos = fsize;
    if (end_pos >= 128 && memcmp(data + end_pos - 128, "TAG", 3) == 0) end_pos -= 128;
    if (end_pos >= 32 && memcmp(data + end_pos - 32, "APETAGEX", 8) == 0) {
        uint32_t ape_sz = (uint32_t)data[end_pos-20] | ((uint32_t)data[end_pos-19]<<8) |
                          ((uint32_t)data[end_pos-18]<<16) | ((uint32_t)data[end_pos-17]<<24);
        if (end_pos >= (long)ape_sz + 32 && memcmp(data + end_pos - ape_sz - 32, "APETAGEX", 8) == 0)
            end_pos -= (ape_sz + 32);
        else if (end_pos >= (long)ape_sz)
            end_pos -= ape_sz;
    }

    // Find start of MP3 stream and determine canonical (v0, l0, p0, bi0, si0, cm0, is_cbr)
    int v0=0, l0=0, p0=1, bi0=0, si0=0, pad0=0, cm0=0, fs0=0;
    long start_pos = pos;
    while (pos + 4 <= end_pos) {
        if (parse_header(data + pos, &v0, &l0, &p0, &bi0, &si0, &pad0, &cm0, &fs0)) {
            if (is_valid_chain(data, end_pos, pos, v0, l0, si0)) break;
        }
        pos++;
        if (pos - start_pos > 65536) break;
    }
    if (fs0 == 0 || pos + 4 > end_pos || !is_valid_chain(data, end_pos, pos, v0, l0, si0)) { free(data); return -1; }
    uint8_t b3_low6 = data[pos + 3] & 0x3F;

    // Check if CBR or VBR by sampling up to 50 valid frames
    int is_cbr = 1;
    int sample_bi = -1;
    long scan_p = pos;
    int checked = 0;
    int si_len0 = (v0 == 3) ? (cm0 == 3 ? 17 : 32) : (cm0 == 3 ? 9 : 17);
    long xing_off = pos + 4 + (p0 == 0 ? 2 : 0) + si_len0;
    int has_xing = 0;
    if (xing_off + 16 <= pos + fs0 && (memcmp(data + xing_off, "Info", 4) == 0 || memcmp(data + xing_off, "Xing", 4) == 0)) {
        has_xing = 1;
        if (memcmp(data + xing_off, "Xing", 4) == 0) is_cbr = 0;
    }
    while (scan_p + 4 <= end_pos && checked < 50) {
        int v, l, p, bi, si, pad, cm, fs;
        if (!parse_header(data + scan_p, &v, &l, &p, &bi, &si, &pad, &cm, &fs) || v != v0 || l != l0 || si != si0) break;
        if (!(checked == 0 && has_xing)) {
            if (sample_bi == -1) {
                sample_bi = bi;
                cm0 = cm;
                b3_low6 = data[scan_p + 3] & 0x3F;
            } else if (bi != sample_bi) is_cbr = 0;
        }
        scan_p += fs;
        checked++;
    }
    if (sample_bi != -1) bi0 = sample_bi;

    // If Frame 0 at start_pos was zeroed/corrupted (pos > start_pos), step back to start_pos!
    int base_fs0 = calc_fsize(v0, l0, bi0, si0, 0);
    if (is_cbr && pos > start_pos && base_fs0 > 0) {
        long diff = pos - start_pos;
        long k = diff / base_fs0;
        long rem = diff - k * base_fs0;
        if (k >= 1 && k <= 250 && rem >= 0 && rem <= k) {
            pos = start_pos;
        }
    }

    // Pass 1: Walk and repair all frame headers!
    long *frame_offs = malloc((fsize / 24 + 1024) * sizeof(long));
    int *frame_sizes = malloc((fsize / 24 + 1024) * sizeof(int));
    int *frame_corrupt = calloc((fsize / 24 + 1024), sizeof(int));
    int nframes = 0;

    long cur = pos;
    while (cur < end_pos) {
        if (end_pos - cur < 4) break;
        if (memcmp(data + cur, "TAG", 3) == 0 || memcmp(data + cur, "ID3", 3) == 0 ||
            memcmp(data + cur, "APET", 4) == 0 || memcmp(data + cur, "LYRI", 4) == 0) break;
        if (cur > pos) {
            int all_zero = 1;
            for (long z = cur; z < end_pos && z < cur + 512; z++) if (data[z] != 0) { all_zero = 0; break; }
            if (all_zero) {
                int rem_zero = 1;
                for (long z = cur; z < end_pos; z++) if (data[z] != 0) { rem_zero = 0; break; }
                if (rem_zero) break;
            }
        }

        int v, l, p, bi, si, pad, cm, fs;
        int valid = parse_header(data + cur, &v, &l, &p, &bi, &si, &pad, &cm, &fs) &&
                    v == v0 && l == l0 && si == si0 && p == p0 &&
                    (nframes == 0 || cm == cm0) &&
                    (cur + fs <= end_pos) &&
                    (!is_cbr || nframes == 0 || bi == bi0);

        if (valid && is_valid_chain(data, end_pos, cur + fs, v0, l0, si0)) {
            frame_offs[nframes] = cur;
            frame_sizes[nframes] = fs;
            nframes++;
            cur += fs;
            continue;
        }

        int base_fs = calc_fsize(v0, l0, bi0, si0, 0);

        if (is_cbr) {
            int srate = sr_tab[v0][si0];
            int brate = (l0 == 1) ? ((v0 == 3) ? br_tab[0][bi0] : br_tab[1][bi0]) : ((v0 == 3) ? br_tab[2][bi0] : br_tab[1][bi0]);
            int exact_div = (((v0 == 3 || l0 == 2) ? 144000 : 72000) * brate) % srate == 0;
            if (exact_div) {
                uint32_t clean_hdr = make_hdr_u32(v0, l0, p0, bi0, si0, 0, cm0, b3_low6);
                if (cur + base_fs <= end_pos) {
                    if (read_u32(data + cur) != clean_hdr) {
                        write_u32(data + cur, clean_hdr);
                        frame_corrupt[nframes] = 1;
                    }
                    frame_offs[nframes] = cur;
                    frame_sizes[nframes] = base_fs;
                    nframes++;
                    cur += base_fs;
                    continue;
                } else break;
            }
            int found_k = -1, found_p = -1;
            for (int k = 1; k <= 250; k++) {
                long min_pos = cur + (long)k * base_fs;
                if (min_pos > end_pos) break;
                for (int pad_cnt = 0; pad_cnt <= k; pad_cnt++) {
                    long cand = min_pos + pad_cnt;
                    if (cand > end_pos) break;
                    if (cand == end_pos || is_valid_chain(data, end_pos, cand, v0, l0, si0)) {
                        found_k = k;
                        found_p = pad_cnt;
                        break;
                    }
                }
                if (found_k != -1) break;
            }
            if (found_k != -1) {
                int K = found_k, P = found_p;
                int *dp = malloc((K + 1) * (P + 1) * sizeof(int));
                int *parent = malloc((K + 1) * (P + 1) * sizeof(int));
                for (int i = 0; i <= K * (P + 1) + P; i++) dp[i] = 1000000;
                dp[0] = 0;
                for (int step = 0; step < K; step++) {
                    for (int p_used = 0; p_used <= P && p_used <= step; p_used++) {
                        int cur_cost = dp[step * (P + 1) + p_used];
                        if (cur_cost >= 1000000) continue;
                        long off = cur + (long)step * base_fs + p_used;
                        uint32_t actual = read_u32(data + off);
                        if (step - p_used < K - P) {
                            uint32_t h0 = make_hdr_u32(v0, l0, p0, bi0, si0, 0, cm0, b3_low6);
                            int c0 = cur_cost + popcnt32((actual ^ h0) & 0xFFFFFF00);
                            int idx0 = (step + 1) * (P + 1) + p_used;
                            if (c0 < dp[idx0]) { dp[idx0] = c0; parent[idx0] = 0; }
                        }
                        if (p_used + 1 <= P) {
                            uint32_t h1 = make_hdr_u32(v0, l0, p0, bi0, si0, 1, cm0, b3_low6);
                            int c1 = cur_cost + popcnt32((actual ^ h1) & 0xFFFFFF00);
                            int idx1 = (step + 1) * (P + 1) + (p_used + 1);
                            if (c1 < dp[idx1]) { dp[idx1] = c1; parent[idx1] = 1; }
                        }
                    }
                }
                int *pads = malloc(K * sizeof(int));
                int p_cur = P;
                for (int step = K; step >= 1; step--) {
                    int pd = parent[step * (P + 1) + p_cur];
                    pads[step - 1] = pd;
                    p_cur -= pd;
                }
                for (int step = 0; step < K; step++) {
                    int pd = pads[step];
                    uint32_t clean_hdr = make_hdr_u32(v0, l0, p0, bi0, si0, pd, cm0, b3_low6);
                    if (read_u32(data + cur) != clean_hdr) {
                        write_u32(data + cur, clean_hdr);
                        frame_corrupt[nframes] = 1;
                    }
                    frame_offs[nframes] = cur;
                    frame_sizes[nframes] = base_fs + pd;
                    nframes++;
                    cur += base_fs + pd;
                }
                free(dp); free(parent); free(pads);
                continue;
            }
        }

        // For VBR (or fallback if CBR didn't find exact K):
        int best_bi = bi0, best_pad = 0, best_score = 1000000;
        for (int c_bi = (is_cbr ? bi0 : 1); c_bi <= (is_cbr ? bi0 : 14); c_bi++) {
            for (int c_pad = 0; c_pad <= 1; c_pad++) {
                int c_fs = calc_fsize(v0, l0, c_bi, si0, c_pad);
                if (cur + c_fs > end_pos) continue;
                uint32_t exp_h = make_hdr_u32(v0, l0, p0, c_bi, si0, c_pad, cm0, b3_low6);
                int d_cur = popcnt32((read_u32(data + cur) ^ exp_h) & 0xFFFFFE00);
                int d_next = 0;
                if (cur + c_fs + 4 <= end_pos) {
                    uint32_t nh = read_u32(data + cur + c_fs);
                    uint32_t mask = 0xFFFE0C00;
                    uint32_t target = exp_h & mask;
                    d_next = popcnt32((nh ^ target) & mask) * 4;
                    if (is_valid_chain(data, end_pos, cur + c_fs, v0, l0, si0)) d_next = -50;
                }
                int score = d_cur + d_next;
                if (score < best_score) {
                    best_score = score;
                    best_bi = c_bi;
                    best_pad = c_pad;
                }
            }
        }
        if (best_score == 1000000) break;
        int chosen_fs = calc_fsize(v0, l0, best_bi, si0, best_pad);
        if (cur + chosen_fs > end_pos) break;
        uint32_t clean_hdr = make_hdr_u32(v0, l0, p0, best_bi, si0, best_pad, cm0, b3_low6);
        if (read_u32(data + cur) != clean_hdr) {
            write_u32(data + cur, clean_hdr);
            frame_corrupt[nframes] = 1;
        }
        frame_offs[nframes] = cur;
        frame_sizes[nframes] = chosen_fs;
        nframes++;
        cur += chosen_fs;
    }

    // Handle any remaining bytes in [cur, end_pos) that aren't a metadata tag
    while (cur < end_pos) {
        if (end_pos - cur >= 4 && (memcmp(data + cur, "TAG", 3) == 0 || memcmp(data + cur, "ID3", 3) == 0 ||
            memcmp(data + cur, "APET", 4) == 0 || memcmp(data + cur, "LYRI", 4) == 0))
            break;
        long rem = end_pos - cur;
        if (rem == 128 && end_pos == fsize) {
            memcpy(data + cur, "TAG", 3);
            end_pos -= 128;
            break;
        }
        int hdr_len = 4 + (p0 == 0 ? 2 : 0);
        int si_len = (v0 == 3) ? (cm0 == 3 ? 17 : 32) : (cm0 == 3 ? 9 : 17);
        int base_fs = calc_fsize(v0, l0, bi0, si0, 0);
        if (rem >= base_fs && base_fs > 0) {
            memset(data + cur, 0, base_fs);
            write_u32(data + cur, make_hdr_u32(v0, l0, p0, bi0, si0, 0, cm0, b3_low6));
            cur += base_fs;
        } else if (rem >= hdr_len + si_len) {
            memset(data + cur, 0, rem);
            write_u32(data + cur, make_hdr_u32(v0, l0, p0, bi0, si0, 0, cm0, b3_low6));
            cur = end_pos;
        } else if (nframes > 0 && bi0 < 14) {
            long prev = frame_offs[nframes - 1];
            memset(data + prev + 4, 0, end_pos - (prev + 4));
            write_u32(data + prev, make_hdr_u32(v0, l0, p0, bi0 + 1, si0, 0, cm0, b3_low6));
            nframes--;
            cur = end_pos;
        } else {
            break;
        }
    }

    // Pass 2a: Repair Side Information & Bit Reservoir across all frames!
    long reservoir_bytes = 0;
    for (int i = 0; i < nframes; i++) {
        if (i == 0 && has_xing) {
            reservoir_bytes = 0;
            continue;
        }
        long f_off = frame_offs[i];
        int fs = frame_sizes[i];
        int v, l, p, bi, si_idx, pad, cm, dummy_fs;
        if (!parse_header(data + f_off, &v, &l, &p, &bi, &si_idx, &pad, &cm, &dummy_fs)) continue;
        if (l != 1) continue; // Only Layer III has side info

        int hdr_len = 4 + (p == 0 ? 2 : 0);
        int si_len = (v == 3) ? (cm == 3 ? 17 : 32) : (cm == 3 ? 9 : 17);
        if (f_off + hdr_len + si_len > end_pos) continue;

        uint8_t *si = data + f_off + hdr_len;
        int ngr = (v == 3) ? 2 : 1;
        int nch = (cm == 3) ? 1 : 2;
        int mdb_bits = (v == 3) ? 9 : 8;
        int priv_bits = (v == 3) ? (nch == 1 ? 5 : 3) : (nch == 1 ? 1 : 2);
        int scfsi_bits = (v == 3) ? (nch * 4) : 0;
        int gr_bits = (v == 3) ? 59 : 63;

        uint32_t mdb = get_bits(si, 0, mdb_bits);
        // 1. Ensure mdb <= reservoir_bytes!
        if ((long)mdb > reservoir_bytes) {
            frame_corrupt[i] = 1;
            for (int b = mdb_bits - 1; b >= 0; b--) {
                if ((mdb & (1 << b)) && (long)(mdb & ~(1 << b)) <= reservoir_bytes) {
                    mdb &= ~(1 << b);
                    break;
                }
            }
            if ((long)mdb > reservoir_bytes) mdb = reservoir_bytes;
            set_bits(si, 0, mdb_bits, mdb);
        }

        int near_corrupt = 0;
        for (int k = (i >= 15 ? i - 15 : 0); k <= (i + 15 < nframes ? i + 15 : nframes - 1); k++) {
            if (frame_corrupt[k]) { near_corrupt = 1; break; }
        }

        int base_bp = mdb_bits + priv_bits + scfsi_bits;
        long total_p23 = 0;
        for (int gr = 0; gr < ngr; gr++) {
            for (int ch = 0; ch < nch; ch++) {
                int bp = base_bp + (gr * nch + ch) * gr_bits;
                uint32_t p23 = get_bits(si, bp, 12);
                uint32_t bv  = get_bits(si, bp + 12, 9);
                uint32_t gg  = get_bits(si, bp + 21, 8);
                int sfc_len  = (v == 3) ? 4 : 9;
                uint32_t ws  = get_bits(si, bp + 29 + sfc_len, 1);

                // 2. Fix bv > 288
                if (bv > 288) {
                    frame_corrupt[i] = 1;
                    near_corrupt = 1;
                    bv &= 0xFF;
                    if (bv > 288) bv &= 0x7F;
                    set_bits(si, bp + 12, 9, bv);
                }

                // 3. Fix illegal ws=1, bt=0 or reserved Huffman table_select (4 or 14)
                int tbl_bp = bp + 30 + sfc_len;
                if (ws) {
                    uint32_t bt = get_bits(si, tbl_bp, 2);
                    if (bt == 0) {
                        frame_corrupt[i] = 1;
                        near_corrupt = 1;
                        ws = 0;
                        set_bits(si, bp + 29 + sfc_len, 1, 0);
                    } else {
                        for (int t = 0; t < 2; t++) {
                            uint32_t ts = get_bits(si, tbl_bp + 3 + t * 5, 5);
                            if (ts == 4) set_bits(si, tbl_bp + 3 + t * 5, 5, 0);
                            if (ts == 14) set_bits(si, tbl_bp + 3 + t * 5, 5, 15);
                        }
                    }
                }
                if (!ws) {
                    for (int t = 0; t < 3; t++) {
                        uint32_t ts = get_bits(si, tbl_bp + t * 5, 5);
                        if (ts == 4) set_bits(si, tbl_bp + t * 5, 5, 0);
                        if (ts == 14) set_bits(si, tbl_bp + t * 5, 5, 15);
                    }
                }

                // 4. If near_corrupt, clamp extreme global_gain spikes (> 200)
                if (near_corrupt && gg > 200 && gg != 210) {
                    gg &= ~0x40;
                    set_bits(si, bp + 21, 8, gg);
                }

                total_p23 += p23;
            }
        }

        // 5. Ensure total_p23 <= avail_bits!
        long main_data_slots = fs - hdr_len - si_len;
        long avail_bits = (mdb + main_data_slots) * 8;
        if (total_p23 > avail_bits) {
            frame_corrupt[i] = 1;
            // First zero out any suspect silent granule (gg == 210 or bv == 0) that has non-zero p23
            for (int gc = 0; gc < ngr * nch; gc++) {
                int bp = base_bp + gc * gr_bits;
                uint32_t p23 = get_bits(si, bp, 12);
                uint32_t bv  = get_bits(si, bp + 12, 9);
                uint32_t gg  = get_bits(si, bp + 21, 8);
                if (p23 > 0 && (gg == 210 || bv == 0)) {
                    set_bits(si, bp, 12, 0);
                    set_bits(si, bp + 12, 9, 0);
                    total_p23 -= p23;
                }
            }
            while (total_p23 > avail_bits) {
                int max_idx = 0;
                uint32_t max_p23 = 0;
                for (int gc = 0; gc < ngr * nch; gc++) {
                    int bp = base_bp + gc * gr_bits;
                    uint32_t p23 = get_bits(si, bp, 12);
                    if (p23 > max_p23) { max_p23 = p23; max_idx = gc; }
                }
                if (max_p23 == 0) break;
                int bp = base_bp + max_idx * gr_bits;
                uint32_t new_p23 = max_p23;
                for (int b = 11; b >= 0; b--) {
                    if (new_p23 & (1 << b)) {
                        new_p23 &= ~(1 << b);
                        break;
                    }
                }
                set_bits(si, bp, 12, new_p23);
                total_p23 -= (max_p23 - new_p23);
            }
        }

        reservoir_bytes = (avail_bits - total_p23) / 8;
        if (reservoir_bytes < 0) reservoir_bytes = 0;
        long max_res = (v == 3) ? 511 : 255;
        if (reservoir_bytes > max_res) reservoir_bytes = max_res;
    }

    // Pass 2b: Acoustic De-Clicking (Silent-Granule / Local Median gg, bv, and Huffman linbits consistency)
    if (l0 == 1 && nframes > 4) {
        for (int i = 1; i < nframes; i++) {
            long f_off = frame_offs[i];
            int v, l, p, bi, si_idx, pad, cm, dummy_fs;
            if (!parse_header(data + f_off, &v, &l, &p, &bi, &si_idx, &pad, &cm, &dummy_fs) || l != 1) continue;
            int ngr = (v == 3) ? 2 : 1;
            int nch = (cm == 3) ? 1 : 2;
            int hdr_len = 4 + (p == 0 ? 2 : 0);
            int si_len = (v == 3) ? (cm == 3 ? 17 : 32) : (cm == 3 ? 9 : 17);
            if (f_off + hdr_len + si_len > end_pos) continue;
            int mdb_bits = (v == 3) ? 9 : 8;
            int priv_bits = (v == 3) ? (nch == 1 ? 5 : 3) : (nch == 1 ? 1 : 2);
            int scfsi_bits = (v == 3) ? (nch * 4) : 0;
            int gr_bits = (v == 3) ? 59 : 63;
            int sfc_len = (v == 3) ? 4 : 9;
            int base_bp = mdb_bits + priv_bits + scfsi_bits;
            uint8_t *si = data + f_off + hdr_len;

            for (int ch = 0; ch < nch; ch++) {
                // Sample window [i-20 .. i+20] for this channel
                int silent_cnt = 0, total_cnt = 0;
                int gg_sum = 0, bv_sum = 0, act_cnt = 0;
                for (int k = (i >= 20 ? i - 20 : 1); k <= (i + 20 < nframes ? i + 20 : nframes - 1); k++) {
                    if (k == i) continue;
                    long k_off = frame_offs[k];
                    int kv, kl, kp, kbi, ksi_idx, kpad, kcm, kfs;
                    if (!parse_header(data + k_off, &kv, &kl, &kp, &kbi, &ksi_idx, &kpad, &kcm, &kfs) || kcm != cm || kv != v) continue;
                    if (k_off + hdr_len + si_len > end_pos) continue;
                    const uint8_t *ksi = data + k_off + hdr_len;
                    for (int gr = 0; gr < ngr; gr++) {
                        int bp = base_bp + (gr * nch + ch) * gr_bits;
                        uint32_t p23 = get_bits(ksi, bp, 12);
                        uint32_t bv  = get_bits(ksi, bp + 12, 9);
                        uint32_t gg  = get_bits(ksi, bp + 21, 8);
                        total_cnt++;
                        if (p23 == 0 && bv == 0) {
                            silent_cnt++;
                        } else if (p23 > 32 && gg < 195 && bv <= 200) {
                            gg_sum += gg;
                            bv_sum += bv;
                            act_cnt++;
                        }
                    }
                }
                int ch_is_silent = (total_cnt >= 10 && silent_cnt * 10 >= total_cnt * 8);
                int avg_gg = (act_cnt > 0) ? (gg_sum / act_cnt) : 160;
                int avg_bv = (act_cnt > 0) ? (bv_sum / act_cnt) : 66;

                for (int gr = 0; gr < ngr; gr++) {
                    int bp = base_bp + (gr * nch + ch) * gr_bits;
                    uint32_t p23 = get_bits(si, bp, 12);
                    uint32_t bv  = get_bits(si, bp + 12, 9);
                    uint32_t gg  = get_bits(si, bp + 21, 8);
                    uint32_t ws  = get_bits(si, bp + 29 + sfc_len, 1);

                    if (ch_is_silent && (p23 != 0 || bv != 0 || gg != 210)) {
                        if (ch == 1 && nch == 2) {
                            int bp0 = base_bp + (gr * nch + 0) * gr_bits;
                            uint32_t p23_0 = get_bits(si, bp0, 12);
                            if (p23_0 > 0 && p23_0 < 300 && p23 >= 512) {
                                set_bits(si, bp0, 12, p23_0 + 512);
                            }
                        }
                        set_bits(si, bp, 12, 0);
                        set_bits(si, bp + 12, 9, 0);
                        set_bits(si, bp + 21, 8, 210);
                        set_bits(si, bp + 29, sfc_len, 0);
                        set_bits(si, bp + 29 + sfc_len, 1, 0);
                        continue;
                    }

                    // In ANY channel, if p23 == 0 or gg == 210 with tiny/power-of-2 p23, bv and p23 must be 0!
                    if (p23 == 0 || (gg == 210 && (bv == 0 || p23 <= 64 || (p23 & (p23 - 1)) == 0))) {
                        if (p23 != 0 || bv != 0) {
                            set_bits(si, bp, 12, 0);
                            set_bits(si, bp + 12, 9, 0);
                            set_bits(si, bp + 29, sfc_len, 0);
                            set_bits(si, bp + 29 + sfc_len, 1, 0);
                        }
                        continue;
                    }

                    // Active granule: clamp extreme bv and gg outliers and high-linbits Huffman table_select
                    if (act_cnt >= 4) {
                        if ((int)bv > avg_bv + 40 || ((int)bv < avg_bv - 40 && avg_bv > 50)) {
                            set_bits(si, bp + 12, 9, avg_bv);
                        }
                        if ((int)gg > avg_gg + 10 || (int)gg < avg_gg - 25) {
                            set_bits(si, bp + 21, 8, avg_gg);
                        }
                    }
                    int tbl_bp = bp + 30 + sfc_len;
                    if (ws && avg_bv >= 50) {
                        uint32_t t0 = get_bits(si, tbl_bp, 5);
                        uint32_t t1 = get_bits(si, tbl_bp + 5, 5);
                        if (t0 >= 15 && t1 >= 15) {
                            ws = 0;
                            set_bits(si, bp + 29 + sfc_len, 1, 0);
                        }
                    }
                    if (!ws) {
                        for (int t = 0; t < 3; t++) {
                            uint32_t ts = get_bits(si, tbl_bp + t * 5, 5);
                            if (ts >= 28 || ts == 22 || ts == 23) {
                                set_bits(si, tbl_bp + t * 5, 5, ts & ~4);
                            } else if (ts < 15 && avg_bv >= 50) {
                                set_bits(si, tbl_bp + t * 5, 5, 24 | (ts & 3));
                            }
                        }
                        if (v == 3 && avg_bv >= 50) {
                            uint32_t r0 = get_bits(si, tbl_bp + 15, 4);
                            uint32_t r1 = get_bits(si, tbl_bp + 19, 3);
                            if (r0 != 4 && r0 != 5 && r0 != 6 && r0 != 7) set_bits(si, tbl_bp + 15, 4, 4);
                            if (r1 != 5 && r1 != 4 && r1 != 3) set_bits(si, tbl_bp + 19, 3, 5);
                            set_bits(si, tbl_bp + 22, 1, 0);
                            set_bits(si, tbl_bp + 23, 1, 0);
                        }
                    }
                }
            }
        }
    }

    // Pass 3: If Frame 0 has a Lavc / LAME Info CRC, update it so the repaired file has a valid CRC!
    if (has_xing && g_update_crc) {
        uint32_t flags = read_u32(data + xing_off + 4);
        long p = xing_off + 8;
        uint32_t xing_bytes = 0;
        if (flags & 1) p += 4;
        if (flags & 2) { xing_bytes = read_u32(data + p); p += 4; }
        if (flags & 4) p += 100;
        if (flags & 8) p += 4;
        if (p + 36 <= pos + fs0 && (memcmp(data + p, "Lavc", 4) == 0 || memcmp(data + p, "LAME", 4) == 0)) {
            uint16_t old_crc = ((uint16_t)data[p+32] << 8) | data[p+33];
            if (old_crc != 0 && pos + fs0 < end_pos) {
                const AVCRC *crc_tbl = av_crc_get_table(AV_CRC_16_ANSI_LE);
                long crc_len = end_pos - (pos + fs0);
                if (xing_bytes > (uint32_t)fs0 && pos + xing_bytes <= fsize) {
                    crc_len = xing_bytes - fs0;
                }
                uint16_t new_crc = av_crc(crc_tbl, 0, data + pos + fs0, crc_len);
                data[p+32] = (new_crc >> 8) & 0xFF;
                data[p+33] = new_crc & 0xFF;
            }
        }
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) { free(data); free(frame_offs); free(frame_sizes); free(frame_corrupt); return -1; }
    fwrite(data, 1, fsize, out);
    fclose(out);

    if (has_stat) {
        if (chown(out_path, st.st_uid, st.st_gid) != 0) { /* ignore non-root chown warning */ }
        struct timespec ts[2] = { st.st_atim, st.st_mtim };
        utimensat(AT_FDCWD, out_path, ts, 0);
    }

    free(data); free(frame_offs); free(frame_sizes); free(frame_corrupt);
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <input.mp3> <output.mp3>     Repair a single MP3 file to <output.mp3>\n"
        "  %s --in-place <file.mp3>        Repair a single MP3 file in-place (preserves mtime & uid:gid)\n"
        "  %s --batch <filelist.txt>       Repair all MP3 files listed in <filelist.txt> in parallel (OpenMP)\n"
        "\n"
        "Options (may appear anywhere):\n"
        "  --update-crc   Also rewrite the embedded LAME/Lavc audio CRC-16 so it matches the repaired data.\n"
        "                 Off by default: the original CRC is the only ground-truth checksum in the file.\n",
        prog, prog, prog);
}

int main(int argc, char **argv) {
    {   /* strip global options */
        int w = 1;
        for (int r = 1; r < argc; r++) {
            if (strcmp(argv[r], "--update-crc") == 0) g_update_crc = 1;
            else argv[w++] = argv[r];
        }
        argc = w;
    }
    if (argc < 3 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return (argc >= 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) ? 0 : 1;
    }

    if (strcmp(argv[1], "--in-place") == 0) {
        return repair_file(argv[2], argv[2]) == 0 ? 0 : 1;
    }

    if (strcmp(argv[1], "--batch") == 0) {
        FILE *fl = fopen(argv[2], "r");
        if (!fl) {
            perror(argv[2]);
            return 1;
        }
        char **paths = NULL;
        size_t n = 0, cap = 0;
        char line[4096];
        while (fgets(line, sizeof(line), fl)) {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
            if (!len) continue;
            if (n == cap) {
                cap = cap ? cap * 2 : 1024;
                paths = realloc(paths, cap * sizeof(char *));
            }
            paths[n++] = strdup(line);
        }
        fclose(fl);

        int ok_cnt = 0, fail_cnt = 0;
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:ok_cnt,fail_cnt)
        for (size_t i = 0; i < n; i++) {
            if (repair_file(paths[i], paths[i]) == 0) ok_cnt++;
            else fail_cnt++;
            free(paths[i]);
        }
        free(paths);
        printf("Batch repair finished: ok=%d fail=%d total=%zu\n", ok_cnt, fail_cnt, n);
        return fail_cnt == 0 ? 0 : 1;
    }

    return repair_file(argv[1], argv[2]) == 0 ? 0 : 1;
}
