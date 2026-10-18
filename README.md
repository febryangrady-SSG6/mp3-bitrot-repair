# mp3-bitrot-repair

Detect and **contain** bit-rot damage in MP3 files (SD cards, cheap flash drives, aging NAND) so they decode
cleanly again instead of glitching, screeching, or stopping. Two small C tools, no GUI, one dependency
(`libavutil` for the CRC table).

This project started as me helping my mom to fix her broken SD Card containing thousands of islamic audio contained within MP3 Files (lossy btw). All of the code is written by AI, as I cannot analyze the bit to know X bit is the problem. This thing is made on my off hours for like 4 hrs. Feel free to modify and criticize this. Also I will never update this by writing the code myself, I cannot even understand those magic numbers.

> [!IMPORTANT]
> **This is damage containment, not lossless recovery.** It makes a corrupted MP3 *decode without errors and
> without loud digital pops*. It **cannot** bring back audio bits that were physically flipped inside the
> compressed payload. If you have a clean copy of the file, use the clean copy. See [Limitations](#limitations).

## Contents
- [What problem this solves](#what-problem-this-solves)
- [Capabilities](#capabilities)
- [Limitations](#limitations)
- [What to expect (measured)](#what-to-expect-measured)
- [Build](#build)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Recommended workflow](#recommended-workflow)
- [Contributing / open problems](#contributing--open-problems)
- [License](#license)

## What problem this solves

Flash media slowly loses charge. A few flipped bits inside an MP3 hit one of four places, with very
different consequences:

| Where the flipped bit lands | What you hear / what the decoder does | Fixable here? |
| :-- | :-- | :-- |
| 4-byte **frame header** | Decoder loses sync; drop-outs, skipped seconds, `Header missing` | **Yes** (reconstructed) |
| **Side info** (`main_data_begin`, `big_values`, `part2_3_length`, `global_gain`, `table_select`, ...) | `invalid new backstep`, `big_values too big`, `overread`, or **full-scale clicks/static** | **Mostly** (rules + local statistics) |
| **Main data** (Huffman-coded spectrum) | A short garbled warble / wrong coefficients, no decoder error | **No** (the information is gone) |
| **ID3v2 tag / cover art** | Broken cover art can make `ffmpeg` fail on the video stream | **Partly** (neutralised, not restored) |

The author built this after a ~15 GB lecture/recitation archive on an SD card turned out to have 591 of
9,629 files (6.14%) damaged.

## Capabilities

**`scan_mp3`**: parallel (OpenMP) structural scanner. Reports files that show any of:
- frame headers that do not chain (desynchronised / corrupted),
- illegal Layer III side info (`big_values > 288`, `window_switching_flag=1` with `block_type=0`),
- bit-reservoir violations (`main_data_begin` larger than the reservoir; `sum(part2_3_length)` larger than the bits available in the frame),
- LAME/Lavc `Info`/`Xing` **audio CRC-16** mismatch (only when the file has that tag).

**`repair_mp3`**: four passes, all in memory, then one write:

| Pass | What it does |
| :-- | :-- |
| **0. ID3v2 / start of stream** | Validates embedded PNG cover art (chunk bounds + CRC-32); if broken, renames the `APIC` frame to `TXXX` so decoders ignore it. Recovers a zeroed Frame 0 directly after the ID3v2 tag. |
| **1. Frame headers** | Rebuilds corrupted 4-byte headers and exact frame boundaries. CBR: Viterbi/DP over the padding bit to pick the lowest Hamming-distance header sequence between two healthy frame chains. VBR: scores candidate `(bitrate, padding)` pairs by current + next header. Handles `TAG`/APE/`ID3` footers and truncated end-of-file stubs. |
| **2a. Side info / bit reservoir** | Clears flipped high bits so `main_data_begin <= reservoir`, `big_values <= 288`; fixes illegal `block_type`; replaces reserved Huffman tables (4, 14); clamps absurd `global_gain` near damaged frames; trims `part2_3_length` until it fits the frame (zeroing suspicious silent granules first). |
| **Frame Repeat Concealment (`--repeat-last`)** | Replaces damaged clusters with the preceding clean frame's payload (`mdb = 0`, slot-padded bitstream) instead of synthesizing/capping audio parameters over corrupted main data. Completely eliminates audio glitches without reservoir drift. |
| **Silence Concealment (`--mute`)** | Replaces damaged clusters with silent frames (`big_values = 0`, zeroed slot payload, balanced bit-reservoir consumption). Smoothly fades volume in and out purely via `global_gain` (default: 0.5s, configurable via `--fade`) to avoid abrupt stops. |
| **3. CRC (opt-in)** | `--update-crc` rewrites the embedded audio CRC-16 to match the repaired data. **Off by default** (see Limitations). |

Other behaviour: in-place mode preserves owner (`uid:gid`) and timestamps; `--batch` repairs a file list in
parallel; MPEG-1/2/2.5 Layer III are handled (including MPEG-2.5, e.g. 8/11/12 kHz speech files).

## Limitations

Read these before pointing it at data you care about.

1. **Not lossless.** Bit flips inside the Huffman-coded main data are not detected and not corrected. The
   repaired file *plays*, but the damaged granules still contain wrong audio (typically a brief warble
   lasting a few frames), not the original sound.
2. **Most "repairs" are heuristic, not restorations.** On the only ground-truth sample available
   (`0004.MP3`, 19,385 bytes, where a separately verified bit-exact fix exists): 1,059 bytes were truly
   corrupted; the tool changed 473 bytes, only **33** of which equal the true original value, **440** are
   different from the true original (they are plausible rather than correct side-info values), and **1,002**
   corrupted bytes were left as they were. The result decodes with zero errors, but it is **not** the
   original file.
3. **Passing the scanner is not proof of correctness.** The scanner checks structural invariants. A file can
   pass and still contain damaged audio. In particular, `--update-crc` makes the CRC check pass *by
   construction*, which is why it is opt-in. Without it, a repaired file will keep reporting `crc=1`,
   which is the honest answer when the data is not bit-exact.
4. **In-place modes have no undo.** `--in-place` and `--batch` overwrite the file. **Keep a backup.**
5. **Heuristics assume "normal" audio.** The de-clicker infers what a granule *should* look like from its
   neighbours. It can wrongly flatten genuine abrupt changes (a very loud transient, an intentional silence
   in one channel) when they happen to look like damage. It needs enough healthy neighbouring frames and
   is weak right at the start/end of a file.
6. **Layer III only for side-info repair.** Layer I/II files get header reconstruction but no side-info or
   de-clicking passes. Free-format streams (bitrate index 0) and files that change sample rate or layer
   mid-stream are not supported and will be rejected or left untouched.
7. **Heavy damage defeats header recovery.** The header search looks up to 250 frames (CBR) / 64 KB ahead for
   a healthy chain. If none is found, frame reconstruction stops early and the rest of the file is left
   exactly as it was (it is still written out, unrepaired from that point). If no valid stream start is found
   at all, the tool returns non-zero and writes nothing.
8. **Unverified acoustically.** There is no automated listening test. Validation used `ffmpeg` decode
   errors, a per-granule sample-jump check on one file, and the structural scanner. Treat results on your
   own files as "decodes cleanly", then listen.
9. **Tags/cover art are not restored**, only neutralised. Other ID3 frames corrupted by bit-rot are left alone.
10. **Platform.** Developed on Linux x86-64 with GCC + OpenMP + `libavutil`. Uses POSIX `stat`/`utimensat`/`chown`.
    Other platforms and big-endian machines are untested.
11. **Test coverage is thin.** One ground-truth file plus a 9,629-file corpus checked structurally. No unit tests yet.
12. **Lyrics3 tags confuse `ffmpeg`.** A Lyrics3 block (`LYRICSBEGIN ... LYRICS200`) placed before the ID3v1 tag
    makes `ffmpeg` print `Header missing` once at EOF, because its demuxer cannot skip it. This is not
    corruption: the audio decodes cleanly if you cut the file at the last frame. The tool leaves such tags
    untouched, so the message remains.

## What to expect (measured)

| Measurement | Result |
| :-- | :-- |
| Corpus: damaged files found by `scan_mp3` | 591 / 9,629 (6.14%) |
| Corpus: files repaired in a batch run | 575 / 575, 0 tool failures (16 more were intentionally skipped) |
| Corpus: `scan_mp3` after batch repair (that run used CRC rewriting) | 0 flagged |
| `ffmpeg -v error -vn` on all 575 repaired files | 574 print nothing; 1 prints `Header missing` once at end-of-file (see Limitations #12) |
| Full-scale sample jumps (>25,000 between adjacent samples) in a heavily damaged 9-minute file, before -> after the de-click pass | 27 granules -> 0 |
| Byte-exact recovery on the one ground-truth file | **No** (33 of 1,059 corrupted bytes restored exactly) |

Realistic expectation: files go from "stutters / screeches / won't play / `ffmpeg` errors" to "plays through
with the odd short artefact where main-data bits were lost". Do not expect archival-grade restoration.

## Build

Requires a C compiler with OpenMP and the FFmpeg `libavutil` development headers.

```bash
# Debian/Ubuntu
sudo apt install build-essential libavutil-dev
# Arch/CachyOS
sudo pacman -S base-devel ffmpeg

make            # builds ./scan_mp3 and ./repair_mp3
```

## Usage

```bash
# 1) Scan: build a file list, then scan it (prints only broken files; exit code 1 if any)
find /path/to/music -iname '*.mp3' > filelist.txt
./scan_mp3 filelist.txt
./scan_mp3 --files a.mp3 b.mp3

# 2) Repair one file to a NEW file (safe: original untouched)
./repair_mp3 broken.mp3 repaired.mp3

# 3) Repair in place / in parallel (BACK UP FIRST)
./repair_mp3 --in-place broken.mp3
./repair_mp3 --batch broken_list.txt

# Concealment modes (substitutes corrupted audio instead of in-place repair):
./repair_mp3 --repeat-last broken.mp3 repaired_repeated.mp3   # repeat preceding clean sound
./repair_mp3 --mute broken.mp3 repaired_muted.mp3             # mute damaged sections (default 0.5s volume fade)
./repair_mp3 --mute --fade 1.0 broken.mp3 repaired_muted.mp3  # custom 1.0s fade-in & fade-out on volume
./repair_mp3 --mute --fade 0 broken.mp3 repaired_muted.mp3    # instant mute (no fade)

# Optional: also rewrite the embedded LAME/Lavc audio CRC (makes the CRC check pass; see Limitations #3)
./repair_mp3 --update-crc broken.mp3 repaired.mp3
```

Scanner output (tab-separated):

```
BROKEN  crc=<0|1>  hdr=<n>  si=<n>  backstep=<n>  <path>
```

| Field | Meaning |
| :-- | :-- |
| `crc` | `1` = LAME/Lavc audio CRC-16 does not match |
| `hdr` | frames whose header did not parse/chain |
| `si` | illegal side-info fields found |
| `backstep` | bit-reservoir / granule-length violations |

Verify a repair independently of this project:

```bash
ffmpeg -v error -i repaired.mp3 -f null -     # no output = decodes without errors
```

## How it works

MP3 (Layer III) frames are `header (4 B) | [CRC] | side info (17/32 B) | main data`. Main data is stored in a
**bit reservoir** that can start in an earlier frame, so one damaged side-info field breaks decoding well beyond
its own frame, and one damaged header throws the whole frame chain out of alignment.

- **Headers** carry a lot of redundancy (version/layer/sample-rate are constant for a file; padding follows a
  fixed cadence), which is why they can be reconstructed reliably.
- **Side info** obeys hard invariants (reservoir size, `big_values <= 288`, lengths that must fit the frame)
  and strong statistical regularity between neighbouring granules, which is what passes 2a/2b exploit.
- **Main data** has essentially no redundancy, which is why it cannot be repaired.

## Recommended workflow

1. **Look for a clean copy first**: another backup, a duplicate in another folder, a re-download. Compare with
   checksums. A clean copy always beats this tool (the author found 3 exact duplicates this way).
2. `scan_mp3` to find the damaged files; keep the list.
3. Back up, then `repair_mp3 in.mp3 out.mp3` (not in-place) and listen to the damaged spots.
4. If the file has a LAME/Lavc CRC and you also have a *verified* reference, prefer that reference over
   `--update-crc`.
5. For the media itself: flash that sits unpowered or is old will keep decaying. Copy data to fresh media and
   keep two copies plus checksums (e.g. `sha256sum`, `par2`).

## Contributing / open problems

Issues and PRs welcome. Ideas, roughly by value:

- **CRC-guided bit-exact recovery.** When an `Info`/`Xing` audio CRC exists it is an oracle: search over small
  bit-flip sets in damaged frames until the CRC matches (this is how a bit-exact result can be proven).
- **Decode-and-check main data**: validate Huffman main data with a real decoder and localise flipped bits.
- **Unit tests** and a corpus of synthetic corruption (flip N random bits in a clean MP3, measure recovery).
- Layer II support, free-format streams, mid-stream parameter changes.
- Smarter concealment in the de-clicker (interpolate from neighbours instead of clamping).
- Windows/macOS support; replace the `libavutil` dependency with an embedded CRC-16 table.

## License

MIT, see [LICENSE](LICENSE).
