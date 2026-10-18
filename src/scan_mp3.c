/*
 * scan_mp3.c — Parallel MP3 Bitstream, Side-Information, Bit-Reservoir & LAME CRC-16 Scanner
 *
 * Detects SD card / NAND flash bit-rot and structural corruption in MP3 files:
 *   1. LAME / Lavc Info/Xing Audio CRC-16 mismatches (AV_CRC_16_ANSI_LE)
 *   2. Corrupted or desynchronized 4-byte MPEG frame headers
 *   3. Illegal Layer III side-information fields (big_values > 288, window_switching_flag=1 & block_type=0)
 *   4. Bit-reservoir backstep underflows (main_data_begin > reservoir_bytes) and granule overreads
 *      (sum(part2_3_length) > available_bits)
 *
 * Build:
 *   gcc -O3 -Wall -Wextra -fopenmp src/scan_mp3.c -o scan_mp3 -lavutil
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

typedef struct {
    char *path;
} FileEntry;

static uint32_t get_bits(const uint8_t *buf, int *bit_pos, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++) {
        int byte_idx = (*bit_pos) >> 3;
        int bit_idx = 7 - ((*bit_pos) & 7);
        val = (val << 1) | ((buf[byte_idx] >> bit_idx) & 1);
        (*bit_pos)++;
    }
    return val;
}

static int parse_header(const uint8_t *h, int *ver, int *layer, int *prot, int *br, int *sr, int *ch_mode, int *fsize) {
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
    *ver = v; *layer = l; *prot = p; *br = brate; *sr = srate;
    *ch_mode = (h[3] >> 6) & 3;
    return 1;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <filelist.txt>              Scan all MP3 file paths listed in <filelist.txt> (one per line)\n"
        "  %s --files <a.mp3> [b.mp3 ...] Scan specific MP3 file(s) directly\n"
        "\n"
        "Output format (tab-separated, printed only for corrupted files):\n"
        "  BROKEN\\tcrc=<0|1>\\thdr=<N>\\tsi=<N>\\tbackstep=<N>\\t<filepath>\n",
        prog, prog);
}

static int scan_one_file(const char *path, const AVCRC *crc_tbl) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return 0; }
    uint8_t *data = malloc(fsize);
    if (!data || fread(data, 1, fsize, f) != (size_t)fsize) {
        fclose(f); free(data); return 0;
    }
    fclose(f);

    long pos = 0;
    while (pos + 10 <= fsize && memcmp(data + pos, "ID3", 3) == 0) {
        long sz = ((data[pos+6] & 0x7f) << 21) | ((data[pos+7] & 0x7f) << 14) |
                  ((data[pos+8] & 0x7f) << 7)  | (data[pos+9] & 0x7f);
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

    int v0=0, l0=0, p0=0, br0=0, sr0=0, cm0=0, fs0=0;
    long start_pos = pos;
    while (pos + 4 <= end_pos) {
        if (parse_header(data + pos, &v0, &l0, &p0, &br0, &sr0, &cm0, &fs0)) {
            if (pos + fs0 + 4 <= end_pos) {
                int v1, l1, p1, br1, sr1, cm1, fs1;
                if (parse_header(data + pos + fs0, &v1, &l1, &p1, &br1, &sr1, &cm1, &fs1) && v0 == v1 && l0 == l1 && sr0 == sr1)
                    break;
            } else break;
        }
        pos++;
        if (pos - start_pos > 65536) break;
    }

    int si_len0 = (v0 == 3) ? (cm0 == 3 ? 17 : 32) : (cm0 == 3 ? 9 : 17);
    long xing_off = pos + 4 + (p0 == 0 ? 2 : 0) + si_len0;
    int has_xing = 0;
    uint32_t xing_frames = 0, xing_bytes = 0;
    int has_lame_crc = 0;
    uint16_t expected_audio_crc = 0;

    if (xing_off + 16 <= pos + fs0 && (memcmp(data + xing_off, "Info", 4) == 0 || memcmp(data + xing_off, "Xing", 4) == 0)) {
        has_xing = 1;
        uint32_t flags = ((uint32_t)data[xing_off+4]<<24) | ((uint32_t)data[xing_off+5]<<16) |
                         ((uint32_t)data[xing_off+6]<<8)  | (uint32_t)data[xing_off+7];
        long p = xing_off + 8;
        if (flags & 1) { xing_frames = ((uint32_t)data[p]<<24) | ((uint32_t)data[p+1]<<16) | ((uint32_t)data[p+2]<<8) | data[p+3]; p += 4; }
        if (flags & 2) { xing_bytes  = ((uint32_t)data[p]<<24) | ((uint32_t)data[p+1]<<16) | ((uint32_t)data[p+2]<<8) | data[p+3]; p += 4; }
        if (flags & 4) p += 100;
        if (flags & 8) p += 4;
        (void)xing_frames;
        if (p + 36 <= pos + fs0 && (memcmp(data + p, "Lavc", 4) == 0 || memcmp(data + p, "LAME", 4) == 0)) {
            expected_audio_crc = ((uint16_t)data[p+32] << 8) | data[p+33];
            if (expected_audio_crc != 0) has_lame_crc = 1;
        }
    }

    int crc_mismatch = 0;
    if (has_lame_crc && pos + fs0 < end_pos) {
        uint16_t crc1 = av_crc(crc_tbl, 0, data + pos + fs0, end_pos - (pos + fs0));
        uint16_t crc2 = crc1;
        if (xing_bytes > (uint32_t)fs0 && pos + xing_bytes <= fsize) {
            crc2 = av_crc(crc_tbl, 0, data + pos + fs0, xing_bytes - fs0);
        }
        uint16_t crc3 = av_crc(crc_tbl, 0, data + pos + fs0, fsize - (pos + fs0));
        if (crc1 != expected_audio_crc && crc2 != expected_audio_crc && crc3 != expected_audio_crc) {
            crc_mismatch = 1;
        }
    }

    int bad_headers = 0, bad_sideinfo = 0, bad_backstep = 0, frame_count = 0;
    long reservoir_bytes = 0;
    long cur = pos;
    while (cur < end_pos) {
        if (end_pos - cur < 4) break;
        if (memcmp(data + cur, "TAG", 3) == 0 || memcmp(data + cur, "ID3", 3) == 0 ||
            memcmp(data + cur, "APET", 4) == 0 || memcmp(data + cur, "LYRI", 4) == 0) break;
        int all_zero = 1;
        for (long z = cur; z < end_pos; z++) if (data[z] != 0) { all_zero = 0; break; }
        if (all_zero) break;

        int v, l, p, br, sr, cm, fs;
        if (!parse_header(data + cur, &v, &l, &p, &br, &sr, &cm, &fs) || v != v0 || l != l0 || sr != sr0) {
            bad_headers++;
            int base_fs = sr0 ? (((v0 == 3 ? 144000 : 72000) * br0) / sr0) : 0;
            int tv, tl, tp, tbr, tsr, tcm, tfs;
            if (base_fs > 0 && cur + base_fs + 4 <= end_pos &&
                parse_header(data + cur + base_fs, &tv, &tl, &tp, &tbr, &tsr, &tcm, &tfs) && tv == v0 && tl == l0 && tsr == sr0) {
                cur += base_fs;
            } else if (base_fs > 0 && cur + base_fs + 1 + 4 <= end_pos &&
                       parse_header(data + cur + base_fs + 1, &tv, &tl, &tp, &tbr, &tsr, &tcm, &tfs) && tv == v0 && tl == l0 && tsr == sr0) {
                cur += base_fs + 1;
            } else {
                long next = cur + 1;
                while (next + 4 <= end_pos) {
                    if (parse_header(data + next, &tv, &tl, &tp, &tbr, &tsr, &tcm, &tfs) && tv == v0 && tl == l0 && tsr == sr0) {
                        if (next + tfs + 4 > end_pos) break;
                        int v2, l2, p2, br2, sr2, cm2, fs2;
                        if (parse_header(data + next + tfs, &v2, &l2, &p2, &br2, &sr2, &cm2, &fs2) && v2 == v0 && l2 == l0 && sr2 == sr0)
                            break;
                    }
                    next++;
                }
                cur = next;
            }
            reservoir_bytes = 511;
            frame_count++;
            continue;
        }

        int hdr_len = 4 + (p == 0 ? 2 : 0);
        int si_len = (v == 3) ? (cm == 3 ? 17 : 32) : (cm == 3 ? 9 : 17);
        if (!(frame_count == 0 && has_xing) && l == 1 && cur + hdr_len + si_len <= end_pos) {
            int bp = 0;
            const uint8_t *si = data + cur + hdr_len;
            int ngr = (v == 3) ? 2 : 1;
            int nch = (cm == 3 ? 1 : 2);
            int mdb = get_bits(si, &bp, (v == 3) ? 9 : 8);
            if (mdb > reservoir_bytes + 64) bad_backstep++;
            if (reservoir_bytes < mdb) reservoir_bytes = mdb;
            get_bits(si, &bp, (v == 3) ? (nch == 1 ? 5 : 3) : (nch == 1 ? 1 : 2));
            if (v == 3) get_bits(si, &bp, nch * 4);
            long total_p23_bits = 0;
            for (int gr = 0; gr < ngr; gr++) {
                for (int ch = 0; ch < nch; ch++) {
                    uint32_t p23 = get_bits(si, &bp, 12);
                    uint32_t bv  = get_bits(si, &bp, 9);
                    uint32_t gg  = get_bits(si, &bp, 8);
                    uint32_t sfc = get_bits(si, &bp, (v == 3) ? 4 : 9);
                    uint32_t ws  = get_bits(si, &bp, 1);
                    (void)gg; (void)sfc;
                    total_p23_bits += p23;
                    if (bv > 288) bad_sideinfo++;
                    if (ws) {
                        uint32_t bt = get_bits(si, &bp, 2);
                        if (bt == 0) bad_sideinfo++;
                        get_bits(si, &bp, 20);
                    } else {
                        get_bits(si, &bp, 22);
                    }
                    if (v == 3) get_bits(si, &bp, 1);
                    get_bits(si, &bp, 2);
                }
            }
            long main_data_slots = fs - hdr_len - si_len;
            long avail_bits = (mdb + main_data_slots) * 8;
            if (total_p23_bits > avail_bits) bad_backstep++;
            reservoir_bytes = (avail_bits - total_p23_bits) / 8;
            if (reservoir_bytes < 0) reservoir_bytes = 0;
            if (reservoir_bytes > 511) reservoir_bytes = 511;
        } else {
            reservoir_bytes = 511;
        }
        cur += fs;
        frame_count++;
    }

    int is_broken = (crc_mismatch || bad_headers > 0 || bad_sideinfo > 0 || bad_backstep > 0);
    if (is_broken) {
        #pragma omp critical
        printf("BROKEN\tcrc=%d\thdr=%d\tsi=%d\tbackstep=%d\t%s\n",
               crc_mismatch, bad_headers, bad_sideinfo, bad_backstep, path);
    }
    free(data);
    return is_broken;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    const AVCRC *crc_tbl = av_crc_get_table(AV_CRC_16_ANSI_LE);

    if (strcmp(argv[1], "--files") == 0) {
        int broken = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:broken)
        for (int i = 2; i < argc; i++) {
            broken += scan_one_file(argv[i], crc_tbl);
        }
        return broken > 0 ? 1 : 0;
    }

    FILE *flist = fopen(argv[1], "r");
    if (!flist) {
        perror(argv[1]);
        return 1;
    }
    FileEntry *files = NULL;
    size_t nfiles = 0, cap = 0;
    char line[4096];
    while (fgets(line, sizeof(line), flist)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (!len) continue;
        if (nfiles == cap) {
            cap = cap ? cap * 2 : 16384;
            files = realloc(files, cap * sizeof(FileEntry));
        }
        files[nfiles++].path = strdup(line);
    }
    fclose(flist);

    int broken = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:broken)
    for (size_t idx = 0; idx < nfiles; idx++) {
        broken += scan_one_file(files[idx].path, crc_tbl);
        free(files[idx].path);
    }
    free(files);
    return broken > 0 ? 1 : 0;
}
