from pdf2image import convert_from_path

pages = convert_from_path("texts/pdfs/text.pdf", dpi=150)  # высокое разрешение
for i, page in enumerate(pages):
    page.save(f"texts/pngs/original/page_{i}.png", "PNG")