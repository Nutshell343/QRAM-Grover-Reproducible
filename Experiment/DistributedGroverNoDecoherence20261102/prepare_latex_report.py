"""Export saved data and vector figures into the editable Chinese TeX project.

Standard-library only. Does not overwrite prose, integrate pulses, or run hardware.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "feasibility"
PROJECT = HERE / "latex_report"
PULSE = HERE.parent / "RouterPulseCalibration20261002"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def percent(value, digits=4):
    return f"{100*value:.{digits}f}" + r"\%"


def sci(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def save(name, text):
    (PROJECT / "tables" / name).write_text(text + "\n", encoding="utf8")


def table(name, caption, label, headers, rows, columns=None):
    spec = columns or "l" + "r" * (len(headers) - 1)
    text = [r"\begin{table}[htbp]\centering\small",
            r"\caption{"+caption+r"}\label{"+label+"}",
            r"\begin{tabular}{@{}"+spec+r"@{}}\toprule",
            " & ".join(headers) + r"\\\midrule"]
    text += [" & ".join(map(str, row)) + r"\\" for row in rows]
    text += [r"\bottomrule\end{tabular}\end{table}"]
    save(name, "\n".join(text))


def longtable(name, caption, label, headers, rows, columns):
    spacing = r"\fontsize{9}{10.6}\selectfont\renewcommand{\arraystretch}{1.0}" if name.startswith("native_") else ""
    text = [r"{\small\setlength{\tabcolsep}{4pt}"+spacing, r"\begin{longtable}{@{}"+columns+r"@{}}",
            r"\caption{"+caption+r"}\label{"+label+r"}\\\toprule",
            " & ".join(headers)+r"\\\midrule\endfirsthead",
            r"\multicolumn{"+str(len(headers))+r"}{c}{续表："+caption+r"}\\\toprule",
            " & ".join(headers)+r"\\\midrule\endhead",
            r"\midrule\multicolumn{"+str(len(headers))+r"}{r}{下页续}\endfoot",
            r"\bottomrule\endlastfoot"]
    text += [" & ".join(map(str, row))+r"\\" for row in rows]
    text += [r"\end{longtable}}"]
    save(name, "\n".join(text))


def main():
    for subdir in ("tables", "images", "data", "supplement"):
        (PROJECT / subdir).mkdir(exist_ok=True)
    metrics = read(DATA / "metrics.json")
    checks = read(DATA / "checks.json")
    resources = read(DATA / "resources.json")
    operations = read(DATA / "native_operation_details.json")
    native = read(PULSE / "results/native/native_parameters.json")
    device = read(PULSE / "device.json")
    macro = {"IdealError": sci(checks["ideal_max_matrix_error"])}
    for scheme, prefix in (("r5", "Rfive"), ("r6", "Rsix")):
        macro[prefix+"Grover"] = percent(metrics["Grover_mean"][scheme])
        macro[prefix+"Fe"] = percent(metrics["network"][scheme]["Fe_mean"])
        macro[prefix+"GHZ"] = percent(metrics["GHZ"][scheme]["F"])
    macro.update(RfivePairPop=percent(metrics["local"]["r5_down"]["truth"][3][3]),
                 RfiveNoPairPop=percent(metrics["local"]["r5_without_pair"]["truth"][3][3]),
                 RfiveLocalFe=percent(metrics["local"]["r5_down"]["Fe_legal"]),
                 RfiveNoPairFe=percent(metrics["local"]["r5_without_pair"]["Fe_legal"]))
    save("data.tex", "% Exported from saved coherent data, not measured hardware.\n" +
         "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key,value in macro.items()))
    roles = ("C", "I", "L", "R")
    table("device.tex", "局域四能级参考器件", "tab:device", ["线", "参考标签", "$f_{01}$/GHz", "$\\alpha$/GHz"],
          [[roles[q], device["physical_labels"][q], f"{device['frequencies_ghz'][q]:.3f}",
            f"{device['anharmonicities_ghz'][q]:.3f}"] for q in range(4)], "llrr")
    table("local_results.tex", "局域原始指标；16态重叠与合法四线过程是不同检验", "tab:localresults",
          ["方案/方向", "16态平均重叠", "合法过程$\\Fe$", "01生存率"],
          [[name.replace("_", " ").upper(), percent(metrics["local"][name]["Fmean"]),
            percent(metrics["local"][name]["Fe_legal"]), percent(metrics["local"][name]["survival01"])]
           for name in ("r5_down", "r5_up", "r5_roundtrip", "r6_down", "r6_up", "r6_roundtrip")])
    table("grover.tex", "四目标Grover：条件高值必须与接受率及联合比例一起读", "tab:grover",
          ["方案/目标", "原始成功", "辅助0概率", "条件成功", "联合正确"],
          [[s.upper()+"/"+format(j,"02b"), *[percent(r[k]) for k in
            ("success", "p_aux0", "success_aux0", "joint_correct_aux0")]]
           for s in ("r5", "r6") for j,r in enumerate(metrics["Grover"][s])])
    resource_rows = [
        ["总角色数",17,17], ["router调用",8,8], ["router内部CZ",40,48],
        ["CZ总数", *[resources[f"2_{s}"]["counts"]["CZ"] for s in ("r5","r6")]],
        ["其中跨片CZ",18,18],
        ["物理单门", *[resources[f"2_{s}"]["counts"]["single"] for s in ("r5","r6")]],
        ["CZ层深", *[resources[f"2_{s}"]["CZ_depth"] for s in ("r5","r6")]],
        ["总调度层（含入口等待）", *[resources[f"2_{s}"]["physical_layers"] for s in ("r5","r6")]],
        ["query时长/ns", *[f"{resources[f'2_{s}']['query_ns']:.3f}" for s in ("r5","r6")]],
        ["含扩散总时长/ns", *[f"{resources[f'2_{s}']['total_ns']:.3f}" for s in ("r5","r6")]],
    ]
    table("resources.tex", "两层四存储的一轮Grover；不含真实准备、复位与读出", "tab:resources",
          ["资源", "R5", "R6"], resource_rows)
    qrows = [
        (r"$G(A_0)$", "地址高位到根输入，D恢复"),
        (r"$S_{T_0C_0}$", "高位存入根控制，根输入回0"),
        (r"$G(A_1)$", "低位到根输入，D恢复"),
        (r"根router down", "按根控制把低位送至左右子输入"),
        (r"$S_{T_1C_1}\parallel S_{T_2C_2}$", "保存低位，两输入端口回0"),
        (r"$H_D=Y_{01}^{(D)}/2\to X_{01}^{(D)}$", "将异或读取转为相位形式"),
        (r"$S_{DT_0}$", "数据进入根；跨片"),
        (r"根router down", "选择左右子树"),
        (r"左down $\parallel$ 右down", "数据至对应叶端"),
        (r"$\prod_j\CZ(L_j,M_j)$（四门并行）", "固定访问：只有被寻址叶端携带数据"),
        (r"左up $\parallel$ 右up", "数据返至子输入，叶端恢复"),
        (r"根router up", "数据返回根输入"),
        (r"$S_{DT_0}$", "数据返回QPU；跨片"),
        (r"$S_{T_1C_1}\parallel S_{T_2C_2}$", "取出低位，子控制恢复"),
        (r"根router up", "低位返回根输入"),
        (r"$G(A_1)$", "恢复A1"),
        (r"$S_{T_0C_0}$", "取出高位，根控制恢复"),
        (r"$G(A_0)$", "恢复A0，临时树回零"),
        (r"$H_D=Y_{01}^{(D)}/2\to X_{01}^{(D)}$", "解码为异或query"),
    ]
    longtable("query_sequence.tex", "完整query的宏观时间顺序", "tab:querysequence",
              ["步", "操作（时间由上到下）", "目的与约束"],
              [[j,*row] for j,row in enumerate(qrows,1)],
              r"r p{.44\linewidth} p{.43\linewidth}")
    gate = {"01_half": r"$X_{01}/2$", "01_pi": r"$X_{01}$", "12_pi": r"$X_{12}$",
            "cz": r"$\CZ$", "frame": "$F$", "frame_01": "$F_{01}$", "frame_12": "$F_{12}$"}
    for scheme in ("r5","r6"):
        for direction in ("down","up"):
            item=operations[f"{scheme}_{direction}"]
            longtable(f"native_{scheme}_{direction}.tex",
                      f"{scheme.upper()} {direction}：{item['physical_layers']}层，{item['duration_ns']:.6f} ns",
                      f"tab:{scheme}{direction}",
                      ["序", "层", "起点/ns", "门", "线", "$\\phi_{\\rm nom}$", "$\\phi_{\\rm buf}$", "门长/ns"],
                      [[r["sequence"],r["layer"],f"{r['start_ns']:.3f}",gate[r["gate"]],
                        ",".join(r["roles"]), "---" if r["gate"]=="cz" else f"{r['nominal_phase_rad']:.5f}",
                        "---" if r["buffered_phase_rad"] is None else f"{math.remainder(r['buffered_phase_rad'],2*math.pi):.5f}",
                        f"{r['pulse_duration_ns']:.3f}"] for r in item["records"]], "rrrllrrr")
    table("native_parameters.tex", "单门冻结参数（程序幅度单位换算为MHz）", "tab:singleparams",
          ["线", "门", "T/ns", "A/MHz", "$\\delta$/MHz", "DRAG"],
          [[roles[r["wire"]],gate[r["gate"]],f"{r['duration']:.0f}",f"{1000*r['amplitude']:.6f}",
            f"{1000*r['detuning']:.6f}",f"{r['drag']:.6f}"] for r in native["single"]],"llrrrr")
    # Both short native parameter tables are one input, but remain separate floats.
    tempfile=PROJECT/"tables/native_cz_parameters.tex"
    table(tempfile.name, "CZ冻结参数，三边峰值因子均为1", "tab:czparams",
          ["边", "调谐线", "$f^{\\CZ}$/GHz", "T/ns", "坡/ns"],
          [["--".join(roles[q] for q in r["edge"]),roles[r["tuning_wire"]],f"{r['frequency']:.9f}",
            f"{r['duration']:.6f}",f"{r['ramp']:.6f}"] for r in native["cz"]],"llrrr")
    with (PROJECT/"tables/native_parameters.tex").open("a",encoding="utf8") as f:
        f.write("\\input{tables/native_cz_parameters.tex}\n")
    par={}
    for s in ("r5","r6"):
        for d in ("down","up"):
            par[s,d]=read(PULSE/f"results/lab_equivalent_runs/{s}/{d}/pulse-L4-shots1000/parameters.json")
    table("endpoint_parameters.tex", "端点帧：down放输出，up放输入，角度单位rad", "tab:endpoint",
          ["方案/方向", "$\\phi_C$", "$\\phi_L$", "$\\phi_R$"],
          [[s.upper()+" "+d,*[f"{par[s,d]['angles'][q]:.6f}" for q in ("c","l","r")]]
           for s in ("r5","r6") for d in ("down","up")])
    matrices=[]
    for d in ("down","up"):
        matrix=par["r5",d]["paired"]["matrix"]
        body=[]
        for row in matrix:
            body.append(" & ".join(f"{z[0]:.6f}{z[1]:+.6f}i" for z in row))
        matrices.append(r"\begin{equation}V_{\mathrm{"+d+r"}}=\begin{pmatrix}"+
                        r"\\".join(body)+r"\end{pmatrix}.\end{equation}")
    save("paired_matrices.tex","\n".join(matrices))
    catalog=read(DATA/"figure_catalog.json")
    for entry in catalog:
        source=DATA/"figures"/(entry["name"]+".pdf")
        assert source.exists(),source
        shutil.copy2(source,PROJECT/"images"/source.name)
    core=["metrics.json","resources.json","checks.json","SUMMARY.json","local.json",
          "pulse_convergence.json","scaling.json","ideal_validation.json","pulse_sources.json",
          "manifest.json","circuit_details_manifest.json"]
    for name in core:shutil.copy2(DATA/name,PROJECT/"data"/name)
    for name in ("schedule_2_r5.json","schedule_2_r6.json","native_operation_details.json",
                 "native_layers_r5.md","native_layers_r6.md"):
        if name.endswith(".md"):
            text=(DATA/name).read_text(encoding="utf8")
            text=text.replace("../CIRCUIT_DETAILS.md", "../main.tex").replace("返回线路详解", "返回TeX主文档")
            text=text.replace("主文档第12节", "TeX附录的逐门表")
            (PROJECT/"supplement"/name).write_text(text,encoding="utf8")
        else:
            shutil.copy2(DATA/name,PROJECT/"supplement"/name)
    shutil.copy2(PULSE/"device.json",PROJECT/"data/device.json")
    shutil.copy2(PULSE/"results/native/native_parameters.json",PROJECT/"data/native_parameters.json")
    for (s,d),parvalue in par.items():
        (PROJECT/"data"/f"parameters_{s}_{d}.json").write_text(json.dumps(parvalue,ensure_ascii=False,indent=2),encoding="utf8")
    files=[p for sub in ("images","tables","data","supplement") for p in (PROJECT/sub).rglob("*") if p.is_file()]
    manifest={"scope":"Native Chinese LaTeX report, existing coherent data only; no reoptimization or hardware execution",
              "figure_count":len(catalog),"files_sha256":{p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (PROJECT/"asset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    print("Exported",len(catalog),"vector figures and",len(list((PROJECT/'tables').glob('*.tex'))),"native TeX data tables.")


if __name__=="__main__":main()
