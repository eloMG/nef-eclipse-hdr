# NEF Eclipse HDR

NEF Eclipse HDR develops bracketed Nikon NEF eclipse photographs into linear
RGB, registers each frame with **x/y translation only**, and produces either:

- one untone-mapped 32-bit floating-point TIFF for each bracket; or
- every registered frame as a separate 16-bit linear TIFF.

The original NEF files are opened read-only and are never modified. The
pipeline intentionally performs no tone mapping, sharpening, denoising,
deghosting, contrast enhancement, lens correction, or creative color
adjustment.

> **Experimental software:** this is a specialized eclipse-processing tool, not
> a general RAW converter or a calibrated scientific-radiometry pipeline. Check
> its diagnostics and inspect every result before relying on it.

## Scope and limitations

- **Nikon NEF input only.** The program discovers files with a `.nef` extension
  directly inside the input directory. It does not scan subdirectories or
  accept other RAW formats, and it does not verify the camera maker or model.
  Support for a particular camera body and NEF compression mode also depends on
  the installed rawpy/LibRaw version.
- **Five-frame brackets are the tested workflow.** Five is the default, although
  `--group-size` accepts any value of at least two. The positional middle frame
  is the reference, so an odd group size is preferable.
- **Bracket boundaries are not detected.** Files are ordered by capture time
  when every timestamp is available, otherwise by natural filename order, and
  then divided into consecutive fixed-size groups. Keep unrelated NEFs out of
  the input directory. An incomplete trailing group is ignored with a warning.
- **A consistent camera setup is assumed.** All frames must develop to the same
  dimensions. White balance, orientation, and sensor white level from the
  reference frame are applied to the whole bracket.
- **Alignment is translation-only.** Rotation, scale, perspective, affine or
  local warping, and lens distortion are not corrected. Moving clouds,
  foreground objects, or changing eclipse features may therefore leave ghosts
  or other merge artifacts.
- **Automatic localization is Sun-specific.** A dominant horizon, cloud edge,
  or other bright feature can be selected instead. Use `--roi X,Y,W,H` when the
  automatic crop is wrong.
- **HDR merging depends on exposure metadata.** Shutter speed and ISO must be
  valid. If every aperture is missing, the program assumes it stayed constant.
  If only some are missing, it fills them from the bracket median only when the
  known apertures agree within 2%; otherwise it rejects the bracket.
- **The HDR values are relative.** They are exposure-normalized, camera-developed
  linear RGB values referenced to the middle frame—not absolute scene radiance
  or calibrated solar measurements.
- **The float TIFF is unprofiled.** It records linear-sRGB/BT.709 primaries in
  its description, but downstream software must interpret the samples through
  an appropriate linear-RGB color-management workflow.
- **The documented environment is 64-bit Windows with Python 3.11 or 3.12.**
  Automated validation currently covers Windows with Python 3.12; other
  platforms and interpreter versions are unvalidated.
- **Full-resolution processing is storage-intensive.** Temporary arrays and
  uncompressed output can require several gigabytes.
- **JSON sidecars contain resolved local paths.** Review them before sharing if
  directory names are sensitive.

This is an independent project and is not affiliated with or endorsed by Nikon.
“Nikon” and “NEF” are used only to identify the supported input format.

## Installation

Use 64-bit Python 3.11 or 3.12. A virtual environment keeps the imaging
dependencies isolated:

```powershell
cd C:\path\to\repository
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`rawpy` publishes Windows wheels containing LibRaw, so a separate LibRaw
installation is normally unnecessary.

## Validate one bracket first

Put one chronological bracket directly in the input directory. The safest
first run processes group 0 only and saves visual diagnostics:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\HDR" `
  --single-bracket `
  --alignment-preview `
  --debug
```

Inspect:

- the printed `dx`/`dy` offsets, which are the shifts applied to each frame;
- `diagnostics\*_alignment_preview.png`, with unaligned crops above and aligned
  crops below;
- `diagnostics\*_edge_overlay.png`, where reference edges are red, moving edges
  are cyan, and close overlap appears pale or white;
- `diagnostics\*_HDR_preview.png`, after a successful merge, as a
  display-stretched inspection image that does not alter the TIFF master;
- `*_HDR.json`, which always records the alignment details and, after a
  successful merge, the exposure factors and fallback counts; and
- after a successful merge, the solar limb and corona in the `*_HDR.tif` master
  at high zoom.

The default `--on-suspicious skip` policy writes a report and diagnostics but no
HDR master when the alignment checks are not convincing. If visual inspection
shows that the offsets are correct, rerun that bracket deliberately with
`--on-suspicious continue`. Add `--overwrite` only when replacing output that
already exists. The override is recorded in the JSON sidecar.

If automatic localization selects the wrong feature, provide one shared crop in
native-image pixels:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py "D:\...\NEF" "D:\...\HDR" `
  --single-bracket --roi 3100,1800,2200,2200 --alignment-preview
```

## Export aligned frames without merging

Use `--aligned-only` to write every registered frame and skip the HDR merge.
For example, this processes only zero-based group 3:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\Aligned" `
  --start-group 3 `
  --single-bracket `
  --aligned-only `
  --alignment-preview
```

With the default group size, group 3 contains the 16th through 20th files in
sorted order. Its files are written as:

```text
<output-dir>\aligned\<reference-stem>\00_<source-stem>_aligned_linear.tif
<output-dir>\aligned\<reference-stem>\01_<source-stem>_aligned_linear.tif
...
<output-dir>\aligned\<reference-stem>\<reference-stem>_aligned.json
```

Each TIFF is a full-size 16-bit linear RGB image at that source frame's original
exposure. Bilinear translation places it in the reference frame's coordinates;
pixels outside the shifted source footprint are black. This mode does not need
the shutter, ISO, or aperture metadata required by the HDR merge.

`--keep-intermediates` serves a different purpose: it retains the very large
developed images *before* alignment under `output\intermediates`. It can be
combined with `--aligned-only` when both unaligned and aligned files are needed.

## Batch processing and grouping

Once a single bracket has been checked, process every complete group:

```powershell
.\.venv\Scripts\python.exe eclipse_hdr.py `
  "D:\Photos\Eclipse\NEF" `
  "D:\Photos\Eclipse\HDR" `
  --group-size 5
```

Group numbers are zero-based and both range endpoints are inclusive:

```powershell
# Process groups 12 through 20.
.\.venv\Scripts\python.exe eclipse_hdr.py "D:\...\NEF" "D:\...\HDR" `
  --start-group 12 --end-group 20
```

Common controls:

| Option | Purpose |
| --- | --- |
| `--group-size N` | Consecutive files per bracket; default `5` |
| `--start-group N`, `--end-group N` | Select an inclusive zero-based group range |
| `--single-bracket` | Process only `--start-group` |
| `--max-shift PX` | Bound both translation axes; default `20` |
| `--registration-crop PX` | Minimum native-pixel automatic crop size |
| `--roi X,Y,W,H` | Override automatic Sun localization |
| `--max-gap-seconds S` | Warn about unexpectedly slow burst timing |
| `--alignment-preview` | Save visual alignment diagnostics |
| `--aligned-only` | Export registered frames without merging |
| `--keep-intermediates` | Retain unaligned developed TIFFs |
| `--on-suspicious {skip,continue,error}` | Choose how failed alignment checks are handled |
| `--overwrite` | Replace existing outputs |
| `--debug` | Enable verbose logging and diagnostics |

Run `python eclipse_hdr.py --help` for the complete, authoritative option list.
After argument parsing, the exit status is `1` if any selected group fails;
otherwise it is `2` if any group is skipped as suspicious, or `0` if all groups
complete. An interruption returns `130`; invalid command syntax also returns `2`.

## Processing model

### Linear RAW development

The reference frame supplies one white-balance vector, one LibRaw orientation
code, and one sensor saturation level for the bracket. Development uses linear
output (`gamma=(1, 1)`), LibRaw's linear black-to-white scaling with a
reference-fixed sensor white point, 16-bit linear-sRGB primaries, AHD demosaicing,
and no automatic brightening, image-dependent maximum adjustment, denoising,
median filtering, or highlight reconstruction.

When compatible sensor metadata is available, the program records CFA sites at
their per-channel white levels before demosaicing and expands that mask over the
demosaic footprint. Registration uses the mask when suitable, and the merge
rejects those clipped measurements. Postprocessed RGB clipping detection is the
fallback.

### Translation-only registration

The only estimated mapping is:

```text
x' = x + dx
y' = y + dy
```

The program locates persistent foreground signal, selects a shared crop, and
compares three exposure-tolerant image representations. It combines independent
shift candidates, checks the selected offsets against the configured bounds and
burst-motion model, and records disagreements in the diagnostics. The final RGB
image is resampled once with bilinear interpolation and constant borders;
nothing wraps around an edge.

### Relative linear HDR merge

For frame `i`, relative exposure is calculated from metadata:

```text
q_i = shutter_seconds * ISO / aperture^2
relative_i = q_i / q_reference
normalized_i = aligned_linear_RGB_i / relative_i
```

The merge weights the normalized measurements according to the original linear
sample level. It suppresses black-level/noise-dominated samples, rolls off near
clipping, rejects known saturated sensor sites, and uses one scalar weight per
RGB triplet to avoid channel-dependent color seams:

```text
HDR = sum(weight_i * normalized_i) / sum(weight_i)
```

If no ideal sample exists, an all-saturated pixel uses the least-exposed frame
as a lower bound and an all-dark pixel uses the highest-exposure frame. Counts
are stored in the JSON sidecar. The result is not tone mapped or normalized to
the range `[0, 1]`.

## Output and color management

HDR masters are single-page, contiguous, uncompressed RGB float32 TIFFs. Values
above `1.0` are preserved. The files deliberately contain no ordinary sRGB ICC
profile, because that would describe a nonlinear transfer curve rather than the
stored linear samples.

Confirm that the first master opens as a 32-bit linear/HDR document in your
editor, that values above `1.0` remain recoverable, and that any assigned profile
describes linear sRGB/scRGB rather than transfer-encoded sRGB. Affinity Photo's
documentation covers
[32-bit HDR editing](https://affinity.help/photo2/English.lproj/pages/HDR/hdr_editing.html)
and [supported formats](https://affinity.help/photo2/English.lproj/pages/Appendix/fileformat.html).
Editor interoperability and color-profile assignment are not automated by this
project.

## Tests

Install the development dependency and run the synthetic test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover translation direction and subpixel accuracy, exposure changes,
candidate consensus, clipped-feature masking, ROI handling, motion checks,
relative exposure normalization, merge fallbacks, non-wrapping borders,
grouping, RAW-development parameter locking, aligned-only output, and float-TIFF
values above `1.0`.

The automated suite uses synthetic or mocked image data. The pipeline has also
been manually checked end to end on two five-frame Nikon NEF eclipse brackets,
but those source photographs are not distributed with this repository.
