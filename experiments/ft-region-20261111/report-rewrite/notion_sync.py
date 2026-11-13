"""Publish reviewed Markdown and local image files through the ntn CLI.

Back up old content, upload real image blocks, replace the existing pages,
and preserve any existing 用户手记 section by block ID. No new pages.
"""
import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

REPORT = "4178b24d-37dc-81a5-93c5-e7abd698670a"
DAILY = "4178b24d-37dc-816f-9f76-ff093f332de1"


def api(path, method="GET", body=None):
    command = ["ntn", "api", path, "-X", method]
    tmp = None
    if body is not None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(body, tmp, ensure_ascii=False)
        tmp.close()
        command += ["--data", "@" + tmp.name]
    try:
        p = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"{method} {path}: {p.stderr[:800]}")
        return json.loads(p.stdout)
    finally:
        if tmp:
            Path(tmp.name).unlink()


def children(page):
    all_blocks, cursor = [], None
    while True:
        path = f"v1/blocks/{page}/children"
        command = ["ntn", "api", path, "page_size==100"]
        if cursor:
            command += ["start_cursor==" + cursor]
        p = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
        d = json.loads(p.stdout)
        all_blocks.extend(d["results"])
        if not d["has_more"]:
            return all_blocks
        cursor = d["next_cursor"]


def rich(text):
    result = []
    pattern = r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))"
    for piece in re.split(pattern, text):
        if not piece:
            continue
        annotations = {}
        link = None
        if piece.startswith("**") and piece.endswith("**"):
            piece = piece[2:-2]
            annotations["bold"] = True
        elif piece.startswith("`") and piece.endswith("`"):
            piece = piece[1:-1]
            annotations["code"] = True
        elif piece.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", piece)
            if match:
                piece, link = match.groups()
        for i in range(0, len(piece), 1800):
            item = {"type": "text", "text": {"content": piece[i:i + 1800]}}
            if link:
                item["text"]["link"] = {"url": link}
            if annotations:
                item["annotations"] = annotations
            result.append(item)
    return result


def text_block(kind, text):
    return {"object": "block", "type": kind, kind: {"rich_text": rich(text)}}


def blocks_from_markdown(text, uploads):
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("```"):
            language = line[3:].strip()
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            result.append({"object": "block", "type": "code",
                           "code": {"language": "bash" if language == "bash" else "plain text",
                                    "rich_text": rich("\n".join(code))}})
            continue
        if line.startswith("|"):
            table = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i].strip())
                i += 1
            cells = [[s.strip() for s in t.strip("|").split("|")] for t in table]
            cells = [r for r in cells if not all(re.fullmatch(r"[: -]+", c) for c in r)]
            width = len(cells[0])
            assert all(len(r) == width for r in cells)
            result.append({"object": "block", "type": "table", "table": {
                "table_width": width, "has_column_header": True, "has_row_header": False,
                "children": [{"object": "block", "type": "table_row",
                              "table_row": {"cells": [rich(c) for c in row]}} for row in cells]}})
            continue
        image = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image:
            caption, path = image.groups()
            upload_id = uploads[Path(path).name]["id"]
            result.append({"object": "block", "type": "image", "image": {
                "type": "file_upload", "file_upload": {"id": upload_id}, "caption": rich(caption)}})
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            result.append(text_block(f"heading_{len(heading[1])}", heading[2]))
            continue
        if line.startswith("- "):
            result.append(text_block("bulleted_list_item", line[2:]))
            continue
        result.append(text_block("paragraph", line))
    return result


def upload(path, receipt, max_size):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.name in receipt and receipt[path.name]["sha256"] == sha:
        return receipt[path.name]
    assert path.stat().st_size <= max_size, path
    d = api("v1/file_uploads", "POST", {"mode": "single_part", "filename": path.name,
            "content_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
    p = subprocess.run(["ntn", "api", f"v1/file_uploads/{d['id']}/send", "-X", "POST",
                        "--file", str(path)], stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, check=True)
    result = json.loads(p.stdout)
    assert result["status"] == "uploaded"
    item = {"id": result["id"], "sha256": sha, "bytes": path.stat().st_size}
    receipt[path.name] = item
    return item


def backup(page, folder):
    top = children(page)
    data = {"page": api(f"v1/pages/{page}"), "blocks": top, "nested": {}}
    for b in top:
        if b.get("has_children"):
            data["nested"][b["id"]] = children(b["id"])
    target = folder / f"{page}-before.json"
    if target.exists():
        target = folder / f"{page}-resume-{int(time.time())}.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return top


def replace(page, new_blocks, old_blocks, protected_ids):
    previous = None
    added = []
    for start in range(0, len(new_blocks), 60):
        before_ids = {b["id"] for b in children(page)}
        position = {"type": "start"} if previous is None else {
            "type": "after_block", "after_block": {"id": previous}}
        api(f"v1/blocks/{page}/children", "PATCH",
            {"children": new_blocks[start:start + 60], "position": position})
        # Some deployed API versions return sibling blocks and paginate the
        # response. Re-read the parent's direct children and identify new IDs.
        created = [b for b in children(page) if b["id"] not in before_ids]
        assert len(created) == len(new_blocks[start:start + 60])
        added.extend(created)
        previous = added[-1]["id"]
    assert len(added) == len(new_blocks)
    for b in old_blocks:
        if b["id"] not in protected_ids:
            api(f"v1/blocks/{b['id']}", "DELETE")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--status", choices=["已完成", "进行中", "遇到问题"], required=True)
    args = ap.parse_args()
    out = args.output
    archive = out / "notion"
    archive.mkdir(exist_ok=True)
    receipt_path = archive / "uploads.json"
    uploads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    me = api("v1/users/me")
    max_size = me["bot"]["workspace_limits"]["max_file_upload_size_in_bytes"]
    old_report = backup(REPORT, archive)
    old_daily = backup(DAILY, archive)
    note_index = next((i for i, b in enumerate(old_daily) if b["type"] == "heading_2" and
                      "".join(t.get("plain_text", "") for t in b["heading_2"]["rich_text"]) == "用户手记"), None)
    protected_ids = {b["id"] for b in old_daily[note_index:]} if note_index is not None else set()
    paths = sorted((out / "figures").glob("*.png"))
    paths += [out / "ft-region-evidence.zip", out / "report.md", out / "daily.md", out / "versions.json"]
    for path in paths:
        upload(path, uploads, max_size)
        receipt_path.write_text(json.dumps(uploads, ensure_ascii=False, indent=2))
        print("uploaded", path.name, flush=True)
    report_text = (out / "report.md").read_text()
    daily_text = (out / "daily.md").read_text()
    if protected_ids:
        daily_text = daily_text.split("## 用户手记")[0].rstrip()
    report_blocks = blocks_from_markdown(report_text, uploads)
    daily_blocks = blocks_from_markdown(daily_text, uploads)
    for path in paths[-4:]:
        report_blocks.append({"object": "block", "type": "file", "file": {
            "type": "file_upload", "file_upload": {"id": uploads[path.name]["id"]},
            "caption": rich({"ft-region-evidence.zip": "原始结果快照、统计表、审计回执、绘图代码与图表 PDF",
                             "report.md": "本版完整报告 Markdown", "daily.md": "2026-11-13 日报 Markdown",
                             "versions.json": "实际运行版本与后处理版本清单"}[path.name])}})
    added_report = replace(REPORT, report_blocks, old_report, set())
    added_daily = replace(DAILY, daily_blocks, old_daily, protected_ids)
    api(f"v1/pages/{REPORT}", "PATCH", {"properties": {
        "标题": {"title": rich("FT 选区如何影响波前恢复：窗宽、交集边界与中心误差（run-01）")},
        "实验时间": {"date": {"start": "2026-11-13"}},
        "执行结果": {"select": {"name": args.status}},
    }})
    api(f"v1/pages/{DAILY}", "PATCH", {"properties": {
        "标题": {"title": rich("2026-11-13｜FT 选区：完整性复核、边界腐蚀收益与中心实验缺口")},
        "日期": {"date": {"start": "2026-11-13"}},
    }})
    actual_report = children(REPORT)
    actual_daily = children(DAILY)
    assert len(actual_report) == len(report_blocks)
    assert len(actual_daily) == len(daily_blocks) + len(protected_ids)
    assert sum(b["type"] == "image" for b in actual_report) == 10
    assert sum(b["type"] == "image" for b in actual_daily) == 2
    assert actual_daily[-1]["type"] == "heading_2" or protected_ids
    assert protected_ids <= {b["id"] for b in actual_daily}
    image_hashes = []
    for b in actual_report:
        if b["type"] != "image":
            continue
        media = b["image"]
        url = media["file"]["url"]
        data = urllib.request.urlopen(url).read()
        sha = hashlib.sha256(data).hexdigest()
        assert sha in {v["sha256"] for k, v in uploads.items() if k.endswith(".png")}
        image_hashes.append(sha)
    for page, name in [(REPORT, "report"), (DAILY, "daily")]:
        p = subprocess.run(["ntn", "pages", "get", page], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, check=True)
        (archive / f"{name}-readback.md").write_text(p.stdout)
    receipt = {"report_id": REPORT, "daily_id": DAILY, "report_blocks": len(actual_report),
               "daily_blocks": len(actual_daily), "report_images": 10, "daily_images": 2,
               "verified_image_hashes": image_hashes,
               "protected_user_note_block_ids": sorted(protected_ids),
               "report_properties": api(f"v1/pages/{REPORT}")["properties"],
               "daily_properties": api(f"v1/pages/{DAILY}")["properties"]}
    (archive / "sync-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if not k.endswith("properties")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
