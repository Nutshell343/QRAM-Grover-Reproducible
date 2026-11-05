"""Build Markdown independently of PDF rendering and numerical simulation."""
from pathlib import Path


def markdown(manifest, prefix):
    md = ["# " + manifest["title"], "",
          "> 只研究理想原生门与相干脉冲过程；不附加耗散、随机门事件、准备/读出错误或有限shots重建。", "",
          "完整门序列、相位约定、冻结参数及QRAM查询步骤见[线路细节与可行性说明](../CIRCUIT_DETAILS.md)。"
          "完整逐层调度：[R5](../feasibility/native_layers_r5.md)、[R6](../feasibility/native_layers_r6.md)。", ""]
    for page in manifest["pages"]:
        md += ["## " + page["title"], ""]
        for text in page["paragraphs"]:
            md += [text, ""]
        if page["equation"]:
            md += [f'![公式]({prefix}/{page["equation"]}.png)', ""]
        if page["figure"]:
            md += [f'![数据图]({prefix}/{page["figure"]}.png)', page["caption"], ""]
        if page["headers"]:
            md += ["| " + " | ".join(page["headers"]) + " |",
                   "| " + " | ".join(["---"] * len(page["headers"])) + " |"]
            md += ["| " + " | ".join(row) + " |" for row in page["rows"]] + [""]
    return "\n".join(md)


def write_markdown_reports(directory, manifest):
    directory = Path(directory)
    for subdirectory, prefix in (("feasibility", "figures"), ("results", "../feasibility/figures")):
        (directory / subdirectory / "REPORT.md").write_text(markdown(manifest, prefix), encoding="utf-8")
