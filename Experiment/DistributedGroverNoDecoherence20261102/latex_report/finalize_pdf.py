"""Preserve searchable Chinese when a Tectonic build omits Fandol ToUnicode.

The native TeX pages, equations, vectors, and content streams are unchanged.
Only missing open-font CID-to-Unicode maps are attached to PDF font objects.
Requires pypdf. Run after XeLaTeX/Tectonic; a normal mapped font is left alone.
"""
from pathlib import Path
import hashlib
import json
import unicodedata
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

HERE = Path(__file__).resolve().parent


def cmap(name, mapping):
    def text(code):
        value = chr(code)
        # Fandol aliases some common Han glyphs to Kangxi radical code points.
        if 0x2E80 <= code <= 0x2FDF or 0xF900 <= code <= 0xFAFF:
            value = unicodedata.normalize("NFKC", value)
        return value.encode("utf-16-be").hex().upper()
    entries = sorted((int(cid), text(code)) for cid, code in mapping.items())
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
             f"/CMapName /{name}-Unicode def", "/CMapType 2 def",
             "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange"]
    for start in range(0, len(entries), 100):
        group = entries[start:start+100]
        lines += [f"{len(group)} beginbfchar"]
        lines += [f"<{cid:04X}> <{unicode}>" for cid, unicode in group]
        lines += ["endbfchar"]
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return "\n".join(lines).encode("ascii")


def main():
    source = HERE / "main.pdf"
    output = HERE / "线路可行性_中文LaTeX报告.pdf"
    newest_tex = max(path.stat().st_mtime for path in HERE.rglob("*.tex"))
    if source.stat().st_mtime + 1 < newest_tex:
        raise RuntimeError("main.pdf predates the TeX sources; compile successfully before finalizing.")
    mapping = json.loads((HERE / "support/fandol_cid_unicode.json").read_text(encoding="utf8"))
    reader = PdfReader(source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    visited = set()
    fixed = []
    for page in writer.pages:
        for reference in page.get("/Resources", {}).get("/Font", {}).values():
            font = reference.get_object()
            identity = id(font)
            if identity in visited:
                continue
            visited.add(identity)
            base = str(font.get("/BaseFont", ""))
            if "/ToUnicode" in font:
                continue
            name = next((key for key in mapping if key in base), None)
            if name:
                stream = DecodedStreamObject()
                stream.set_data(cmap(name, mapping[name]))
                font[NameObject("/ToUnicode")] = writer._add_object(stream.flate_encode())
                fixed.append(base)
    writer.add_metadata({"/Title": "从部分受控交换到QRAM查询：线路可行性与相干脉冲验证",
                         "/Author": "TCG", "/Subject": "原生中文LaTeX、冻结相干仿真、25张矢量数据图"})
    writer.write(output)
    final = PdfReader(output)
    cover = final.pages[0].extract_text().replace(" ", "").replace("\n", "")
    assert "从部分受控交换到QRAM查询" in cover, cover[:300]
    assert "线路可行性与相干脉冲验证" in cover, cover[:300]
    assert len(final.pages) == len(reader.pages)
    # Unicode metadata must not alter any rendered text/image content stream.
    assert all(a.get_contents().get_data() == b.get_contents().get_data()
               for a,b in zip(reader.pages,final.pages))
    result = {"pages": len(final.pages), "fixed_missing_ToUnicode_fonts": fixed,
              "native_content_streams_unchanged": True, "Chinese_text_copy_verified": True,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (HERE / "pdf_checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(output.name, json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
