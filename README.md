# Eclipse HDR

`eclipse_hdr.py` develops Nikon NEF exposure brackets into linear RGB and aligns
each frame with **x/y translation only**. It can either merge the measurements
into one untone-mapped 32-bit floating-point TIFF per bracket or export all five
aligned linear frames without merging them.

It intentionally performs no tone mapping, sharpening, denoising, deghosting,
contrast enhancement, lens correction, or creative color adjustment. The NEFs
are opened read-only and are never modified.

## Install on Windows

Use 64-bit Python 3.11 or 3.12. A virtual environment keeps the imaging
dependencies isolated:

```powershell
cd C:\path\to\align-raw-eclipse
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`rawpy` publishes Windows wheels containing LibRaw, so a separate LibRaw install
is normally unnecessary. Full-resolution Nikon files need substantial temporary
disk space: five disk-backed uint16 RGB developments plus one float32 HDR array
can occupy several gigabytes. Scratch files are deleted after each group.

## Validate one bracket first

Put one or more chronological sets of five NEFs directly in the input folder.
The safest first run processes group 0 only and saves visual diagnostics:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\HDR" `
  --single-bracket `
  --alignment-preview `
  --debug
```

Inspect these before starting the batch:

- the printed `dx`/`dy` offsets (these are shifts **applied** to each frame);
- `diagnostics\*_alignment_preview.png`, whose upper row is unaligned and lower
  row is aligned;
- `diagnostics\*_edge_overlay.png`, where reference edges are red, moving edges
  are cyan, and correct overlap becomes pale/white;
- `diagnostics\*_HDR_preview.png`, a clearly labeled display-stretched crop used
  only to inspect the merged corona/limb (the TIFF master is unchanged);
- `*_HDR.json`, including all exposure metadata, candidates, offsets, physical
  checks, exposure factors, fallback-pixel counts, and warnings;
- the solar limb/corona in the `*_HDR.tif` master at high zoom.

The default policy is `--on-suspicious skip`: a bracket that fails translation
confidence or independent-representation consensus gets a JSON report and
diagnostics but no HDR master. Motion-model deviations remain blocking when the
image evidence is weak; moderate deviations with strong independent consensus
are recorded as advisories, while large discontinuities remain blocking.
After manual inspection, an individual bracket can deliberately be rerun with
`--on-suspicious continue --overwrite`. That override is recorded in its JSON.

If automatic localization chooses the horizon or another bright object, give a
shared native-pixel crop as `--roi X,Y,W,H`, for example:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py "D:\...\NEF" "D:\...\HDR" `
  --single-bracket --roi 3100,1800,2200,2200 --alignment-preview
```

## Export five aligned frames without an HDR merge

Use `--aligned-only` to write the five full-resolution registered images and
skip the HDR merge. To process just group 3 from a folder that also contains
other brackets:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\Aligned" `
  --start-group 3 `
  --single-bracket `
  --aligned-only `
  --max-shift 60 `
  --registration-crop 3000 `
  --alignment-preview `
  --debug
```

Group numbers are zero-based, so group 3 is chronological files 16 through 20.
The output for a bracket whose central reference is `20-27-04.58.NEF` is:

```text
<output-dir>\aligned\20-27-04.58\00_20-27-04.01_aligned_linear.tif
<output-dir>\aligned\20-27-04.58\01_20-27-04.30_aligned_linear.tif
<output-dir>\aligned\20-27-04.58\02_20-27-04.58_aligned_linear.tif
<output-dir>\aligned\20-27-04.58\03_20-27-04.86_aligned_linear.tif
<output-dir>\aligned\20-27-04.58\04_20-27-05.16_aligned_linear.tif
<output-dir>\aligned\20-27-04.58\20-27-04.58_aligned.json
```

Each TIFF is full-size, 16-bit linear RGB and keeps that source frame's original
exposure. Bilinear translation places it in the central frame's coordinates;
pixels outside its shifted source footprint are black. This mode performs no
exposure normalization, HDR merge, tone mapping, sharpening, or denoising.

`--keep-intermediates` has a different purpose: it retains the developed images
*before* alignment under `output\intermediates`. It may be used together with
`--aligned-only` when both the unaligned and aligned versions are wanted.

The suspicious-alignment policy still applies. Inspect the diagnostics first;
if the offsets are correct despite a physical-motion warning, deliberately rerun
the same command with `--on-suspicious continue --overwrite`.

## Batch processing

Once the first bracket looks correct:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\HDR" `
  --group-size 5
```

Every consecutive five files form one bracket. Capture timestamps are used for
ordering when every NEF has one; otherwise natural filename order is used and a
warning is emitted. An incomplete trailing group is never merged. Group numbers
are zero-based, and both ends are inclusive:

```powershell
# Process groups 12 through 20.
.\.venv\Scripts\python.exe eclipse_hdr.py "D:\...\NEF" "D:\...\HDR" `
  --start-group 12 --end-group 20
```

Useful controls:

```text
--group-size 5
--max-shift 20
--max-gap-seconds 2
--registration-crop 2048
--roi X,Y,W,H
--alignment-preview
--aligned-only
--keep-intermediates
--on-suspicious {skip,continue,error}
--start-group N --end-group N
--single-bracket
--overwrite
--debug
```

Run `python eclipse_hdr.py --help` for all radiometric and sanity-check options.
`--aligned-only` writes the translated final frames and skips the HDR merge.
`--keep-intermediates` writes very large *unaligned* linear uint16 TIFFs under
`output\intermediates`; otherwise no developed intermediate TIFFs are retained.

## What the pipeline does

### 1. Linear RAW development

The central frame (index `2` in a five-file bracket) supplies one white-balance
vector, one LibRaw orientation code, and one sensor saturation level for every
member. A camera metadata rotation is applied identically to the whole bracket;
it is not an estimated registration parameter.

The important `rawpy.Params` choices are:

- `gamma=(1, 1)`: linear light, not the normal sRGB transfer curve;
- `no_auto_bright=True`: no histogram-dependent brightness correction;
- `adjust_maximum_thr=0.0`: disables LibRaw's image-dependent maximum
  adjustment, which could otherwise scale bracket members differently;
- `output_bps=16`, fixed `user_wb`, fixed `user_sat`, and linear-sRGB primaries;
- `HighlightMode.Ignore`: preserves channel headroom without highlight blending
  or reconstruction;
- AHD demosaicing, with FBDD noise reduction and median filtering disabled.

LibRaw's normal black-to-white scaling remains enabled. It is linear, and its
white point is fixed from the reference; disabling it would also bypass the
white-balance stage. See the current
[`rawpy.Params` API](https://letmaik.github.io/rawpy/api/rawpy.Params.html),
[`RawPy` properties](https://letmaik.github.io/rawpy/api/rawpy.RawPy.html), and
[`rawpy` linear-16-bit example](https://github.com/letmaik/rawpy).

Before demosaicing, the program records CFA sites at their per-channel sensor
white levels and dilates that mask over the demosaic footprint. Registration
uses localized masks to remove clipped/bloomed evidence, while the HDR merge
uses them to reject clipped measurements. The postprocessed `0.98` threshold is
a secondary clipping check.

### 2. Translation-only alignment

No code path estimates rotation, scale, affine, projective, lens, or perspective
parameters. The sole mapping is:

```text
x' = x + dx
y' = y + dy
```

The program locates persistent foreground signal across the five downsampled
frames, expands one shared crop around it, and builds three exposure-tolerant
representations: log-gradient, log-high-pass, and an MTB-like bitmap. Localized
saturated regions are smoothly excluded so their mask boundaries do not become
false features. A bounded normalized cross-correlation searches only the
configured `+/- max-shift` square. `phase_cross_correlation` then refines the
residual on a subpixel grid. Its returned axis order is `(dy, dx)` and is the
shift to apply to the moving frame, as documented by
[`scikit-image`](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation).

Raw correlation scores from the three representations are not compared as if
they shared one scale. Candidate shifts are clustered spatially, each family
gets one vote, and the cluster corroborated by the most plausible independent
families wins. Boundary, low-PSR, coarse-only, and unsupported solutions remain
visible in diagnostics and can still make a bracket suspicious. Constant-motion
and adjacent-continuity deviations can become advisories when they are bounded
and every frame has strong image consensus; large jumps remain blocking. Failure
is never upgraded to a more flexible transform.

The final developed RGB is resampled exactly once with bilinear interpolation,
constant borders, and an explicit validity mask. Nothing wraps around an edge.

### 3. Linear HDR merge

For frame `i`, the relative exposure is computed from metadata as

```text
q_i = shutter_seconds * ISO / aperture^2
relative_i = q_i / q_reference
radiance_i = aligned_linear_RGB_i / relative_i
```

If every aperture is missing it is treated as constant and the cancellation is
reported. Invalid shutter or ISO metadata stops the bracket. If known apertures
vary while another is missing, the bracket is rejected rather than guessed.

Weights are calculated from the original normalized linear measurement, not the
amplified radiance. A smooth low-end ramp suppresses black-level/noise-dominated
samples; a smooth high-end ramp and the sensor mask reject clipping; a signal
factor favors the highest-SNR still-valid exposure. One scalar weight is used
for all three RGB channels to prevent channel-dependent color seams:

```text
HDR = sum(weight_i * radiance_i) / sum(weight_i)
```

If no ideal sample exists, the behavior is explicit: an all-saturated pixel uses
the least-exposed frame as a lower bound, while an all-dark pixel uses the
highest-exposure frame. Their counts are saved in the JSON sidecar. The result
is never tone mapped or normalized to `[0, 1]`.

## Affinity Photo output

The master is a single-page, contiguous RGB, IEEE float32 TIFF written
uncompressed by `tifffile`. Values above `1.0` are preserved; the automated
round-trip test includes values through `8.0`. The TIFF is deliberately
unprofiled because embedding an ordinary nonlinear sRGB ICC profile would
misdescribe linear samples. The TIFF description records linear-sRGB/BT.709
primaries.

For the first real bracket, verify that Affinity Photo opens it as an `RGB/32`
HDR document and that its 32-bit preview exposure control reveals highlight
values above `1`. Assign a genuine linear-sRGB/scRGB profile if your Affinity
workflow requires a profile; do not convert the samples through an ordinary
transfer-encoded sRGB profile. Affinity documents
[`32-bit HDR editing`](https://affinity.help/photo2/English.lproj/pages/HDR/hdr_editing.html)
and its [`supported formats`](https://affinity.help/photo2/English.lproj/pages/Appendix/fileformat.html).

The TIFF structure and unbounded numeric values are tested here, but actual
Affinity interoperability cannot be certified without opening the first master
in your Affinity installation. If that acceptance check exposes a TIFF/profile
problem, OpenEXR is the appropriate next output backend; Affinity explicitly
supports a 32-bit-linear EXR workflow.

## Tests and validation status

Install the test dependency and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The synthetic suite covers fractional translation sign/axis, exposure changes,
candidate-family consensus, a post-totality partial limb with one-sided bloom,
saturation-mask/ROI handling, camera-jitter advisories, clipping and black
fallback behavior, shutter/ISO/aperture normalization, non-wrapping borders,
grouping, rawpy parameter locking, aligned-only 16-bit outputs, and float-TIFF
values above `1`.

A real five-NEF totality/diamond-ring bracket was also run end to end with rawpy
0.27.0 / LibRaw 0.22.1. It contains 8288 x 5520 Nikon frames at 400 mm, ISO 64,
f/7.1, and 1/250 through 1/40 second. Validation confirmed:

- successful read-only NEF development with identical WB/orientation/white level;
- all three registration representations agreeing to about 0.1–0.3 px;
- observed linear brightness ratios of 1.604, 0.634, 1.000, 2.527, and 4.026,
  versus metadata-predicted 1.600, 0.640, 1.000, 2.667, and 4.000;
- a finite 5520 x 8288 x 3 classic TIFF with contiguous float32 RGB, no
  compression, and no 8-bit conversion;
- five full-resolution 5520 x 8288 aligned-only TIFFs with contiguous uint16
  RGB, preserved individual exposures, and the same measured translations;
- visually coincident lunar-limb edges plus retained corona and prominence detail
  in the display-only diagnostics.

That bracket's measured offsets did **not** follow constant motion: the maximum
line-fit residual was 2.675 px and the adjacent second-difference was 6.002 px.
The original validation used an explicit `--on-suspicious continue` override.
The current policy records this kind of motion as an advisory when all frame
translations have strong independent image support; the numeric deviations
remain in the JSON sidecar.

A second real bracket captured after totality contains a much larger clipped
photospheric region covering part of the lunar limb. The old raw-score ranking
selected an unsupported frame-3 boundary result `(dx=+20, dy=+14)`. Saturation-
aware consensus with a constrained ROI now selects `(dx=-1.80, dy=-1.15)`,
corroborated by gradient and high-pass evidence. The full bracket offsets are
approximately `(-0.45,+4.20)`, `(-1.55,+2.30)`, `(0,0)`, `(-1.80,-1.15)`, and
`(-2.95,-2.60)` as `(dx,dy)`, with visually coincident lunar-limb edges in the
diagnostic overlay.

The final remaining acceptance check is opening the generated master in the
user's Affinity Photo installation and confirming its `RGB/32` interpretation
and linear-profile workflow.
