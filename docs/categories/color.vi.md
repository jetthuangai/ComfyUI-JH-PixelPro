# Màu & tông

Nhóm node này lo mọi thứ về màu và độ sáng: khớp ảnh này với ảnh khác, tạo tương phản bằng đường cong, grade bằng bảng kiểu Camera Raw, chọn vùng theo tông hoặc theo màu, và lưu hay áp dụng LUT. Dùng để hoàn thiện một bản retouch, kéo ảnh AI về lại ảnh tham chiếu, hoặc dựng một look có thể lặp lại.

<div class="grid cards" markdown>

-   ![](../assets/nodes/color-matcher.jpg)

    **[Color Matcher](../nodes/color-matcher.md)**

    Kéo màu của ảnh này về ảnh tham chiếu trong không gian LAB.

-   ![](../assets/nodes/color-matcher-region.jpg)

    **[Color Matcher Region](../nodes/color-matcher-region.md)**

    Khớp một vùng mask với vùng khác, kể cả khi ảnh khác kích thước.

-   ![](../assets/nodes/color-lab.jpg)

    **[ColorLab](../nodes/color-lab.md)**

    Basic, HSL, Color Grading và Gray Mix kiểu Camera Raw trong một node.

-   ![](../assets/nodes/shadows-highlights-pro.jpg)

    **[Shadows/Highlights Pro](../nodes/shadows-highlights-pro.md)**

    Nâng vùng tối, kéo lại vùng sáng mà không bị quầng, kiểu Camera Raw hoặc Photoshop.

-   ![](../assets/nodes/auto-tone.jpg)

    **[Auto: Tone & White Balance](../nodes/auto-tone.md)**

    Một cú bấm: phơi sáng, Whites/Blacks, Highlights/Shadows và ám màu, kèm thông số dạng preset ColorLab.

-   ![](../assets/nodes/apply-colorlab-preset.jpg)

    **[Apply ColorLab Preset](../nodes/apply-colorlab-preset.md)**

    Áp look ColorLab lên ảnh khác hoặc cả loạt ảnh, có strength và mask.

-   ![](../assets/nodes/lightroom-preset.jpg)

    **[Lightroom Preset (XMP)](../nodes/lightroom-preset.md)**

    Dùng preset Lightroom / Camera Raw, có lưới ảnh xem trước và báo cáo.

-   ![](../assets/nodes/lightroom-preset-contact-sheet.jpg)

    **[Lightroom Preset Contact Sheet](../nodes/lightroom-preset-contact-sheet.md)**

    So sánh cả bộ preset trên một tấm ảnh.

-   ![](../assets/nodes/clahe-local-contrast.jpg)

    **[CLAHE Local Contrast](../nodes/clahe-local-contrast.md)**

    Local contrast thích ứng chỉ trên độ sáng; màu sắc giữ nguyên.

-   ![](../assets/nodes/tone-curve.jpg)

    **[Tone Curve](../nodes/tone-curve.md)**

    Đường cong tám điểm cho tương phản và cân bằng màu, có preset.

-   ![](../assets/nodes/hue-saturation-range.jpg)

    **[Hue/Saturation per Range](../nodes/hue-saturation-range.md)**

    Đổi hue hoặc độ bão hòa trong một dải màu được chọn.

-   ![](../assets/nodes/luminosity-masking.jpg)

    **[Luminosity Masking](../nodes/luminosity-masking.md)**

    Mask mềm cho vùng bóng, midtone và vùng sáng theo độ sáng ảnh.

-   ![](../assets/nodes/saturation-mask.jpg)

    **[Saturation Mask](../nodes/saturation-mask.md)**

    Tạo mask cho các pixel có độ đậm màu nằm trong một khoảng.

-   ![](../assets/nodes/lut-preset.jpg)

    **[LUT Preset](../nodes/lut-preset.md)**

    Áp một look màu dựng sẵn từ danh sách, có thanh chỉnh cường độ.

-   ![](../assets/nodes/lut-import.jpg)

    **[LUT Import](../nodes/lut-import.md)**

    Nạp LUT `.cube` bất kỳ và áp dụng, có strength và mask tùy chọn.

-   ![](../assets/nodes/lut-export.jpg)

    **[LUT Export](../nodes/lut-export.md)**

    Lưu ảnh chart đã grade thành LUT `.cube` dùng được ở nơi khác.

-   ![](../assets/nodes/hald-identity.jpg)

    **[HALD Identity](../nodes/hald-identity.md)**

    Tạo ảnh chart màu trung tính để bạn grade rồi bake thành LUT.

-   ![](../assets/nodes/tone-match-lut.jpg)

    **[Tone Match LUT](../nodes/tone-match-lut.md)**

    Tự dựng LUT từ màu và tông của một ảnh tham chiếu.

-   ![](../assets/nodes/palette-extract.jpg)

    **[Palette Extract](../nodes/palette-extract.md)**

    Lấy 2–16 màu chủ đạo thành ô màu và mã HEX, kèm phiên bản ảnh màu phẳng (posterize).

</div>

## Chuỗi node thường dùng

- **Khớp rồi hoàn thiện:** Load Image → [Color Matcher](../nodes/color-matcher.md) → [Tone Curve](../nodes/tone-curve.md) → [ColorLab](../nodes/color-lab.md) → Preview Image
- **Chỉ grade vùng bóng:** Load Image → [Luminosity Masking](../nodes/luminosity-masking.md) → dùng `mask_shadows` làm mask cho [Tone Curve](../nodes/tone-curve.md) hoặc [Hue/Saturation per Range](../nodes/hue-saturation-range.md)
- **Bake look thành LUT:** [HALD Identity](../nodes/hald-identity.md) → [Tone Curve](../nodes/tone-curve.md) / [ColorLab](../nodes/color-lab.md) → [LUT Export](../nodes/lut-export.md), sau đó dùng lại bằng [LUT Import](../nodes/lut-import.md)
