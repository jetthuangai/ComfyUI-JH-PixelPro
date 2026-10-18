# Third-party notices

ComfyUI-JH-PixelPro is licensed under Apache-2.0 (see `LICENSE`). Some modules translate parts of third-party code. Those parts stay under their original licence, reproduced below.

## PhotoCraft

- Source: <https://github.com/storytold/photocraft>, commit `ead6d2197f5d0b82d70f3fddb70e472fc27ce577`, dual-licensed MIT OR Apache-2.0. Used here under MIT.
- Translated into:
  - `core/local_laplacian.py`: from `crates/algo/src/camera_raw/local_tone.rs`. Strength knots, thresholds and percentile rules.
  - `core/shadows_highlights_ps.py`: from `crates/algo/src/tone/shadows_highlights.rs`. Beta mask table, curve strength, Color and Midtone tables.

```
MIT License

Copyright (c) 2026 ArtCraft Team and the PhotoCraft contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
