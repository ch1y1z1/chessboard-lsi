"""Publish the reviewed figure revision without replacing report/daily pages."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report-rewrite"))
from notion_sync import api, children, rich, text_block, upload, blocks_from_markdown

REPORT = "4178b24d-37dc-81a5-93c5-e7abd698670a"
OLD_DAILY = "4178b24d-37dc-816f-9f76-ff093f332de1"
PROJECT = "4148b24d-37dc-8116-b810-e4043e450913"
DAILY_SOURCE = "419513fc-9b7d-47a0-bed4-4a2e540f8d7b"
TIMELINE = "4178b24d-37dc-8157-8e16-ca084e310a22"


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def plain(b, field="rich_text"):
    return "".join(x.get("plain_text", x.get("text", {}).get("content", ""))
                   for x in b[b["type"]].get(field, []))


def canonical(b):
    b = copy.deepcopy(b)
    p = b[b["type"]]
    if p.get("type") == "file":
        p["file"] = {"url_path": p["file"]["url"].split("?")[0]}
    return b


def once(path, method, body):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(body, f, ensure_ascii=False); f.flush()
        p = subprocess.run(["ntn", "api", path, "-X", method, "--data", "@"+f.name],
                           stdin=subprocess.DEVNULL, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr[:1000])
    return json.loads(p.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    out = ap.parse_args().output.resolve()
    source = out/"source"
    manifest = json.loads((out/"figure-manifest.json").read_text())
    by_num = {f["number"]: f for f in manifest if f["number"]}
    audit = json.loads((out/"data-audit.json").read_text())
    assert audit["layout_collisions"] == audit["clipped_text"] == 0
    original_report = json.loads((source/"report-blocks.json").read_text())
    original_daily = json.loads((source/"old-daily-blocks.json").read_text())
    versions = json.loads((out/"versions-revision.json").read_text())
    upload_path = out/"uploads.json"
    uploaded = json.loads(upload_path.read_text()) if upload_path.exists() else {}
    max_size = api("v1/users/me")["bot"]["workspace_limits"]["max_file_upload_size_in_bytes"]
    changed = set()

    def send(path):
        r = upload(path, uploaded, max_size)
        save(upload_path, uploaded)
        return r["id"]

    def patch(bid, kind, payload):
        payload = copy.deepcopy(payload)
        if kind in ["image", "file"]:
            # Update-block media schema omits "type"; append-block includes it.
            payload.pop("type", None)
        api("v1/blocks/"+bid, "PATCH", {kind: payload})
        changed.add(bid)

    def append(page, blocks, after, marker):
        before = children(page)
        if any(marker in plain(b) or marker in plain(b, "caption") for b in before):
            return
        ids = {b["id"] for b in before}
        body = {"children": blocks, "position": {
            "type": "after_block", "after_block": {"id": after}}}
        try:
            api("v1/blocks/"+page+"/children", "PATCH", body)
        except RuntimeError:
            observed = children(page)
            added = [b for b in observed if b["id"] not in ids]
            if len(added) != len(blocks):
                save(out/"uncertain-append.json", {"page": page, "body": body, "observed": observed})
                raise
        observed = children(page)
        added = [b for b in observed if b["id"] not in ids]
        assert len(added) == len(blocks), "Inspect page before retrying a partial append"

    def image(num):
        f = by_num[num]
        return {"object": "block", "type": "image", "image": {
            "type": "file_upload", "file_upload": {"id": send(out/f["png"])},
            "caption": rich(f["caption"]) }}

    def guide(num):
        return text_block("paragraph", f"图 {num} 怎么读："+by_num[num]["guide"])

    def prefix(text):
        matches = [b for b in original_report if plain(b).startswith(text)]
        assert len(matches) == 1, text
        return matches[0]

    # Keep the original ten native image IDs; only their image/caption changes.
    old_images = [b for b in original_report if b["type"] == "image"]
    new_numbers = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13]
    assert len(old_images) == len(new_numbers) == 10
    for old, n in zip(old_images, new_numbers):
        patch(old["id"], "image", image(n)["image"])
        append(REPORT, [guide(n)], old["id"], f"图 {n} 怎么读：")
        print(f"Updated report figure {n}", flush=True)
    extra = [(6, "腐蚀并没有缩小"), (9, "圆窗的 clean RMS"),
             (11, "但这里的阈值实现"), (14, "表中单位为")]
    for n, anchor in extra:
        append(REPORT, [image(n), guide(n)], prefix(anchor)["id"], f"图 {n}｜")
        print(f"Inserted report figure {n}", flush=True)
    intro = original_report[0]
    revised_intro = plain(intro) + " 图表于 2026-11-16 返修，原实验日期与数据来源沿用上述记录。"
    patch(intro["id"], "paragraph", {"rich_text": rich(revised_intro)})
    reading = [
        text_block("heading_3", "读图说明"),
        text_block("paragraph", "所有正文图按 16:9 或 4:3 重绘。数据图默认蓝圆为单源、橙菱形为 64 源；掩膜示意和带宽诊断使用各自图例。clean 表示不加测量噪声。标有 10⁻⁴ waves 的轴，刻度 4 表示误差为 4×10⁻⁴ 个波长，数值越低越好；RMS 不去均值、倾斜或离焦。"),
        text_block("paragraph", "比值图的 1 表示参照配置，百分比图的 0 表示无变化。标明“对数刻度”的轴用倍数而非差值读；部分折线图纵轴不从零起画，应按刻度判断变化。误差条只表示五种子的样本标准差，不是置信区间。每张图后分别说明坐标、单位和适用范围，条件数见附件补图 S1。"),
    ]
    anchor = next(b for b in original_report if plain(b).startswith("原结果文件中的"))
    append(REPORT, reading, anchor["id"], "读图说明")
    version_text = (
        "2026-11-16 图表返修使用 figure-revision/redraw.py：只读取保存结果，"
        "核对 700 项统计并重建四幅已有系数的空间误差图；未启动新的传播、标定或优化。"
        f"仓库 chessboard-lsi，完整 commit {versions['commit']}，"
        f"固定 tag {versions['tag']}。"
        f"[提交]({versions['github_commit']}) · [固定标签]({versions['github_tag']})。"
        "新版 PDF、绘图源稿、审计和版本清单见后附返修包；11 月 13 日附件保留为历史快照。"
    )
    append(REPORT, [text_block("paragraph", version_text)],
           prefix("本版后处理使用")["id"], "2026-11-16 图表返修使用")
    # Replace the HTML in its existing final attachment block.
    old_html = next(b for b in original_report if b["type"] == "file" and
                    b["file"].get("name", "").endswith(".html"))
    html = out/"ft-region-overview-revised-20261116.html"
    patch(old_html["id"], "file", {"type": "file_upload",
          "file_upload": {"id": send(html)}, "name": html.name,
          "caption": rich("2026-11-16 图表返修版简版 HTML；五节总览，图表内嵌，可离线打开。")})
    html_intro = next(b for b in original_report if plain(b).startswith("六个图文面板"))
    patch(html_intro["id"], "paragraph", {"rich_text": rich(
        "五节简版总览复用本次重绘的流程、窗宽、腐蚀与方法比较图，并补充坐标含义和中心实验的局限。"
        "单文件可离线打开；正文保留完整图表与统计口径。图表返修版本见上文固定标签。")})
    # Preserve old user-added daily prose and the complete user-notes tail.
    old_di = [b for b in original_daily if b["type"] == "image"]
    for old, n in zip(old_di, [2, 5]):
        block = image(n)["image"]
        block["caption"] = rich(f"日报图（2026-11-16 返修）｜"+by_num[n]["title"]+"；坐标详见关联报告。")
        patch(old["id"], "image", block)
        short = ("横轴为窗口半宽/基频，纵轴为全光瞳 clean RMS，单位 10⁻⁴ waves；虚线为默认 0.40f。"
                 if n == 2 else
                 "横、纵轴均为相对光轴的相机像素坐标；蓝色保留、橙色删除，仅改变拟合掩膜。")
        append(OLD_DAILY, [text_block("paragraph", f"返修图 {n} 读法："+short)],
               old["id"], f"返修图 {n} 读法：")
    note_i = next(i for i, b in enumerate(original_daily) if plain(b) == "用户手记")
    append(OLD_DAILY, [text_block("paragraph",
           "2026-11-16 图表返修：本页两张图已与实验报告同步替换，补充坐标说明。"
           "当日完成的数值工作仍以上文 2026-11-13 记录为准；本次仅修订表达。")],
           original_daily[note_i-1]["id"], "2026-11-16 图表返修：")
    # Query again immediately before creation to avoid a second daily page.
    query = {"filter": {"and": [
        {"property": "日期", "date": {"equals": "2026-11-16"}},
        {"property": "项目", "relation": {"contains": PROJECT}}]}, "page_size": 100}
    existing = api("v1/data_sources/"+DAILY_SOURCE+"/query", "POST", query)["results"]
    assert len(existing) <= 1
    daily_blocks = blocks_from_markdown((out/"daily.md").read_text(), {})
    if existing:
        today = existing[0]["id"]
        current = children(today); save(source/"today-daily-before.json", current)
        ni = next(i for i, b in enumerate(current) if plain(b) == "用户手记")
        assert ni > 0
        append(today, daily_blocks[:-1], current[ni-1]["id"],
               "按用户反馈返修 FT 选区实验报告")
    else:
        body = {"parent": {"type": "data_source_id", "data_source_id": DAILY_SOURCE},
                "properties": {
                    "标题": {"title": rich("2026-11-16｜FT 选区报告图表返修：拆图并补齐坐标解释")},
                    "日期": {"date": {"start": "2026-11-16"}},
                    "项目": {"relation": [{"id": PROJECT}]},
                    "关联实验": {"relation": [{"id": REPORT}]},
                    "关联时间线": {"relation": [{"id": TIMELINE}]}},
                "children": daily_blocks}
        try:
            today = once("v1/pages", "POST", body)["id"]
        except RuntimeError:
            observed = api("v1/data_sources/"+DAILY_SOURCE+"/query", "POST", query)["results"]
            if len(observed) != 1:
                raise
            today = observed[0]["id"]
    save(out/"today-daily.json", {"id": today})
    # Mark old downloadable material as historical instead of presenting old plots as current.
    for b in original_report:
        if b["type"] == "file" and b["file"].get("name") in ["ft-region-evidence.zip", "daily.md", "versions.json"]:
            api("v1/blocks/"+b["id"], "PATCH", {"file": {
                "caption": rich("2026-11-13 历史快照；本次图表返修与新增读图说明见新版附件。")}})
            changed.add(b["id"])
    files = [
        (out/"figures-revised.pdf", "2026-11-16 返修：14 张正文图与补图 S1，文字和坐标可矢量放大。"),
        (out/"versions-revision.json", "本次图表返修的完整 commit、固定标签、命令与来源哈希。"),
    ]
    # Append new attachments before the existing HTML section; HTML stays last.
    prior_versions = next(b for b in original_report if b["type"] == "file" and
                          b["file"].get("name") == "versions.json")
    wanted = []
    for path, caption in files:
        wanted.append({"object": "block", "type": "file", "file": {
            "type": "file_upload", "file_upload": {"id": send(path)},
            "name": path.name, "caption": rich(caption)}})
    append(REPORT, wanted, prior_versions["id"], "2026-11-16 返修：14 张正文图")
    # Read current Markdown before producing the final downloadable source.
    for name, pid in [("report-revised-20261116", REPORT), ("daily-20261116", today)]:
        p = subprocess.run(["ntn", "pages", "get", pid], stdin=subprocess.DEVNULL,
                           text=True, capture_output=True, check=True)
        (out/(name+"-notion-readback.md")).write_text(p.stdout)
        md = p.stdout
        if pid == REPORT:
            def local_image(m):
                number = re.match(r"图 (\d+)｜", m[1])
                if number:
                    return f"![{m[1]}]({by_num[int(number[1])]['png']})"
                return m[0]
            md = re.sub(r"!\[([^]]*)\]\([^)]+\)", local_image, md)
        (out/(name+".md")).write_text(md)
    report_md = out/"report-revised-20261116.md"
    old_md = next(b for b in original_report if b["type"] == "file" and b["file"].get("name") == "report.md")
    patch(old_md["id"], "file", {"type": "file_upload", "file_upload": {"id": send(report_md)},
          "name": report_md.name, "caption": rich("2026-11-16 返修后的实验报告 Markdown，含全部坐标解释。")})
    # Small reproducibility package. Raw run results remain in the original evidence archive.
    package = out/"figure-revision-20261116.zip"
    code = Path(__file__).parent
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [report_md, out/"daily-20261116.md", out/"overview.md", html,
                  out/"figures-revised.pdf", out/"figure-manifest.json",
                  out/"data-audit.json", out/"layout-audit.json", out/"versions-revision.json",
                  out/"html-verification.json"]:
            z.write(p, p.name)
        for p in sorted((out/"figures").glob("*")):
            if p.suffix == ".png": z.write(p, "figures/"+p.name)
        for p in code.glob("*"):
            if p.is_file(): z.write(p, "source/"+p.name)
    assert package.stat().st_size <= max_size
    append(REPORT, [{"object": "block", "type": "file", "file": {
        "type": "file_upload", "file_upload": {"id": send(package)}, "name": package.name,
        "caption": rich("图表返修包：新版 PNG/PDF、报告、简版 HTML、源代码、图数据、审计与版本清单。")}}],
        prior_versions["id"], "图表返修包：")
    report_after, daily_after = children(REPORT), children(OLD_DAILY)
    today_after = children(today)
    for name, before, after in [("report", original_report, report_after),
                                 ("old-daily", original_daily, daily_after)]:
        after_idx = {b["id"]: b for b in after}
        assert [b["id"] for b in after if b["id"] in {x["id"] for x in before}] == [
            b["id"] for b in before]
        for b in before:
            if b["id"] not in changed:
                assert canonical(b) == canonical(after_idx[b["id"]]), b["id"]
        save(out/(name+"-after.json"), after)
    assert daily_after[-1]["id"] == original_daily[-1]["id"]
    assert plain(today_after[-1]) == "用户手记"
    assert sum(b["type"] == "image" for b in report_after) == 14
    assert report_after[-1]["id"] == old_html["id"]
    assert all(any(plain(b).startswith(f"图 {n} 怎么读：") for b in report_after)
               for n in range(1, 15))
    save(out/"today-daily-after.json", today_after)
    receipt = {
        "report": REPORT, "today_daily": today, "old_daily": OLD_DAILY,
        "report_image_count": 14, "all_14_reading_guides_present": True,
        "original_blocks_retained": True,
        "old_user_notes_id": original_daily[-1]["id"],
        "html_block_id_retained": old_html["id"], "changed_original_ids": sorted(changed),
        "uploads": uploaded, "versions": versions,
    }
    save(out/"sync-receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["uploads", "versions", "changed_original_ids"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
