import fitz
import os

pdf_path = os.path.join(os.path.dirname(__file__), "40055-Article Text-44146-1-2-20260314.pdf")
doc = fitz.open(pdf_path)
text = "\n".join([page.get_text() for page in doc])
out_path = os.path.join(os.path.dirname(__file__), "paper_text.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"Done: {len(text)} chars")
