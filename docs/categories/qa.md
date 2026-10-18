# QA

QA nodes measure a result instead of changing it. Add them at the end of a retouch chain to see how much an edit moved the image, to confirm that a change stayed inside the area you masked, or to compare two settings on the same input.

<div class="grid cards" markdown>

-   ![](../assets/nodes/qa-image-diff.jpg)

    **[QA Image Diff](../nodes/qa-image-diff.md)**

    Compares two images with SSIM, PSNR and MAE and draws a heatmap, absolute or SSIM-map difference.

-   ![](../assets/nodes/cull-score.jpg)

    **[Cull Score](../nodes/cull-score.md)**

    Ranks a batch by sharpness, exposure, closed eyes and duplicates; keeps the best.

</div>

For checking a *mask* rather than an image, use [Mask Preview](../nodes/mask-preview.md) from the [Masks](mask.md) group.

## Typical chains

- **Did my retouch only change the skin?** original and retouched image → [QA Image Diff](../nodes/qa-image-diff.md) (add the skin mask to limit the numbers to that area) → read the `changed_pct` and the heatmap.
- **Compare two settings:** run the same photo through two versions of a node → feed both results to [QA Image Diff](../nodes/qa-image-diff.md) in `ssim_map` mode to see where the structure differs.
- **Check a smoothing pass:** [Skin Smoother](../nodes/edge-aware-smoother.md) → [QA Image Diff](../nodes/qa-image-diff.md) against the original, then use [Mask Preview](../nodes/mask-preview.md) on your skin mask to confirm they line up.
