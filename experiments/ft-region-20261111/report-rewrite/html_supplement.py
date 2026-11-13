"""Append the requested HTML supplement, preserving existing report/daily blocks.

Uses the existing ntn helpers. Never executes the whole-page replacement routine.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from notion_sync import REPORT, DAILY, api, children, rich, text_block, upload


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def text(block):
    return "".join(
        item.get("plain_text", item.get("text", {}).get("content", ""))
        for item in block[block["type"]].get("rich_text", [])
    )


def stable(block):
    result = copy.deepcopy(block)
    payload = result[result["type"]]
    # Signed download URLs refresh on reads; these are not content changes.
    if payload.get("type") == "file":
        payload["file"] = {"uploaded_file": True}
    return result


def append_once(page, wanted, position, folder, name):
    before = children(page)
    before_ids = {b["id"] for b in before}
    save(folder / f"{name}-append-before.json", before)
    body = {"children": wanted, "position": position}
    save(folder / f"{name}-append-request.json", body)
    try:
        api(f"v1/blocks/{page}/children", "PATCH", body)
    except RuntimeError:
        # A transport failure can follow a committed write. Inspect first.
        observed = children(page)
        added = [b for b in observed if b["id"] not in before_ids]
        if len(added) != len(wanted):
            save(folder / f"{name}-uncertain-write.json", observed)
            raise
    after = children(page)
    added = [b for b in after if b["id"] not in before_ids]
    assert len(added) == len(wanted), "Read back before retrying a partial write"
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    html = folder / "ft-region-overview.html"
    versions = json.loads((folder / "versions.json").read_text())
    version_url = versions["github_tag"]
    report_before = children(REPORT)
    daily_before = children(DAILY)
    for name, current in [("report", report_before), ("daily", daily_before)]:
        snapshot = folder / f"{name}-before.json"
        if not snapshot.exists():
            save(snapshot, current)
    notes = [b for b in daily_before if text(b) == "用户手记"]
    assert len(notes) == 1
    note_id = notes[0]["id"]
    note_index = next(i for i, b in enumerate(daily_before) if b["id"] == note_id)
    assert note_index > 0
    note_tail = daily_before[note_index:]
    title = "简版 HTML 总览"
    heading_text = (
        "六个图文面板概括关键结果：FT/LM 对比、窗宽响应、交集腐蚀与中心实验的局限。"
        "单文件可离线打开，完整统计与适用范围见正文。"
        f"制作源稿与同步脚本见[固定版本]({version_url})；本次未运行新的数值实验。"
    )
    receipt_path = folder / "uploads.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    existing_html = [
        b for b in report_before
        if b["type"] == "file" and b["file"].get("name") == html.name
    ]
    if not existing_html:
        assert not any(text(b) == title for b in report_before), (
            "A partial supplement exists; inspect its blocks before continuing"
        )
        user = api("v1/users/me")
        max_size = user["bot"]["workspace_limits"]["max_file_upload_size_in_bytes"]
        uploaded = upload(html, receipt, max_size)
        save(receipt_path, receipt)
        wanted = [
            text_block("heading_2", title),
            text_block("paragraph", heading_text),
            {"object": "block", "type": "file", "file": {
                "type": "file_upload", "file_upload": {"id": uploaded["id"]},
                "name": html.name,
                "caption": rich("FT 选区与波前恢复 · run-01 简版；图表已内嵌，可离线打开。"),
            }},
        ]
        append_once(REPORT, wanted, {"type": "end"}, folder, "report")
    else:
        assert len(existing_html) == 1
    daily_note = (
        "按用户要求，在[实验报告末尾]"
        "(https://app.notion.com/p/4178b24d37dc81a593c5e7abd698670a)"
        "补充简版 HTML 总览，含六个图文面板和内嵌图表，可离线打开。"
        "保留正文原有图片，未新增数值实验。"
    )
    # Compare plain text, since response annotations include extra fields.
    daily_plain = text(text_block("paragraph", daily_note))
    if not any(text(b) == daily_plain for b in children(DAILY)):
        append_once(
            DAILY, [text_block("paragraph", daily_note)],
            {"type": "after_block", "after_block": {"id": daily_before[note_index - 1]["id"]}},
            folder, "daily",
        )
    report_after = children(REPORT)
    daily_after = children(DAILY)
    original_report = json.loads((folder / "report-before.json").read_text())
    original_daily = json.loads((folder / "daily-before.json").read_text())
    for old, current in [(original_report, report_after), (original_daily, daily_after)]:
        current_by_id = {b["id"]: b for b in current}
        assert [b["id"] for b in current if b["id"] in {x["id"] for x in old}] == [
            b["id"] for b in old
        ]
        assert all(stable(b) == stable(current_by_id[b["id"]]) for b in old)
    assert report_after[-1]["type"] == "file"
    assert report_after[-1]["file"]["name"] == html.name
    after_note_index = next(i for i, b in enumerate(daily_after) if b["id"] == note_id)
    assert daily_after[after_note_index:] == note_tail
    for name, current in [("report", report_after), ("daily", daily_after)]:
        save(folder / f"{name}-after.json", current)
    result = {
        "report": REPORT, "daily": DAILY,
        "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
        "html_bytes": html.stat().st_size,
        "report_blocks": [len(original_report), len(report_after)],
        "daily_blocks": [len(original_daily), len(daily_after)],
        "report_images_preserved": sum(b["type"] == "image" for b in original_report),
        "user_notes_block_id": note_id,
        "all_original_blocks_preserved": True,
        "attachment_block_id": report_after[-1]["id"],
        "versions": versions,
    }
    save(folder / "sync-receipt.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "versions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
