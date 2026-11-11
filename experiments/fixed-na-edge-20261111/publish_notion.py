"""Archive report via ntn with native tables, frozen files and readback checks."""

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

DATA_SOURCE = "96f383f6-f877-4124-8553-8bf920f3b77c"
DAILY_SOURCE = "419513fc-9b7d-47a0-bed4-4a2e540f8d7b"
PROJECT = "4148b24d-37dc-8116-b810-e4043e450913"
TASK = "fixed-na-edge-20261111"
TITLE = "固定 NA_out 的外缘误差：无分桶曲线、固定面积与高误差像素对照"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("action", choices=("create", "files", "links", "verify"))
    args = parser.parse_args()
    root = args.report.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def read(name):
        return json.loads((root / name).read_text())

    def run(command, text=None):
        result = subprocess.run(command, input=text, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    def api(path, method="GET", body=None, extra=()):
        command = ["ntn", "api", path, "-X", method]
        if body is not None:
            save("last-request.json", body)
            command += ["-d", "@" + str(root / "last-request.json")]
        return run(command + list(extra))

    def children(page):
        blocks, cursor = [], None
        while True:
            params = ["page_size==100"]
            if cursor:
                params.append("start_cursor==" + cursor)
            result = api(f"v1/blocks/{page}/children", extra=params)
            blocks.extend(result["results"])
            if not result["has_more"]:
                return blocks
            cursor = result["next_cursor"]

    def rich(text, url=None):
        item = {"content": str(text)}
        if url:
            item["link"] = {"url": url}
        return [{"type": "text", "text": item}]

    def text_of(block):
        return "".join(x.get("plain_text", x.get("text", {}).get("content", ""))
                       for x in block.get(block["type"], {}).get("rich_text", []))

    def append(page, blocks, after=None):
        body = {"children": blocks}
        if after:
            body["position"] = {"type": "after_block", "after_block": {"id": after}}
        return api(f"v1/blocks/{page}/children", "PATCH", body)

    def create():
        save("experiment-schema-current.json", api(f"v1/data_sources/{DATA_SOURCE}"))
        save("daily-schema-current.json", api(f"v1/data_sources/{DAILY_SOURCE}"))
        daily = api(f"v1/data_sources/{DAILY_SOURCE}/query", "POST", {
            "filter": {"and": [
                {"property": "项目", "relation": {"contains": PROJECT}},
                {"property": "日期", "date": {"equals": "2026-11-11"}},
            ]}, "page_size": 100,
        })
        save("daily-dedup-current.json", daily)
        assert len(daily["results"]) == 1 and not daily["has_more"]
        daily_id = daily["results"][0]["id"]
        save("daily-template-current.json", run(["ntn", "pages", "get", "4148b24d-37dc-8123-962c-d319155d1d58", "--json"]))
        save("daily-before.json", run(["ntn", "pages", "get", daily_id, "--json"]))
        found = api(f"v1/data_sources/{DATA_SOURCE}/query", "POST", {
            "filter": {"and": [
                {"property": "项目", "relation": {"contains": PROJECT}},
                {"property": "任务标识", "rich_text": {"equals": TASK}},
            ]}, "page_size": 100,
        })
        save("report-dedup.json", found)
        assert len(found["results"]) <= 1 and not found["has_more"]
        if found["results"]:
            page = found["results"][0]
        else:
            source = (root / "report.md").read_text()
            source = re.sub(r"\A# .+\n", "", source)
            images, tables = {}, []

            def image_anchor(match):
                path = Path(match.group(2))
                images[path.name] = {"path": str(path), "caption": match.group(1)}
                return f"图：{path.name}"

            source = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_anchor, source)
            lines, output, i = source.splitlines(), [], 0
            while i < len(lines):
                if lines[i].startswith("| "):
                    group = []
                    while i < len(lines) and lines[i].startswith("| "):
                        group.append([x.strip() for x in lines[i].strip().strip("|").split("|")])
                        i += 1
                    assert len(group) >= 3
                    rows = group[:1] + group[2:]
                    tables.append(rows)
                    output.append(f"表：{len(tables)}")
                else:
                    output.append(lines[i])
                    i += 1
            source = "---\ntitle: " + json.dumps(TITLE, ensure_ascii=False) + "\n---\n"
            source += "HTML 概览与完整数据见附件。\n\n" + "\n".join(output)
            (root / "notion-body.md").write_text(source)
            save("images.json", images)
            save("tables.json", tables)
            page = run(["ntn", "pages", "create", "--parent", f"data-source:{DATA_SOURCE}", "--json"], source)
        save("page-created.json", page)
        updated = api(f"v1/pages/{page['id']}", "PATCH", {"properties": {
            "标题": {"title": rich(TITLE)}, "项目": {"relation": [{"id": PROJECT}]},
            "相关日报": {"relation": [{"id": daily_id}]},
            "任务标识": {"rich_text": rich(TASK)}, "类型": {"select": {"name": "分析验证"}},
            "执行结果": {"select": {"name": "已完成"}},
            "实验时间": {"date": {"start": "2026-11-11"}},
        }})
        save("page-properties.json", updated)
        save("daily-id.json", daily_id)
        print("Report:", updated["url"], flush=True)

    def attach(page, path, kind, caption="", after=None):
        journal = read("upload-journal.json") if (root / "upload-journal.json").exists() else {}
        name = path.name
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if name not in journal:
            limit = api("v1/users/me")["bot"]["workspace_limits"]["max_file_upload_size_in_bytes"]
            assert len(data) < limit
            upload = api("v1/file_uploads", "POST", {
                "mode": "single_part", "filename": name,
                "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
            })
            journal[name] = {"path": str(path), "bytes": len(data), "sha256": sha,
                             "kind": kind, "upload": upload}
            save("upload-journal.json", journal)
        entry = journal[name]
        assert entry["sha256"] == sha
        if entry["upload"]["status"] != "uploaded":
            entry["upload"] = api(f"v1/file_uploads/{entry['upload']['id']}/send", "POST", extra=["--file", str(path)])
            assert entry["upload"]["status"] == "uploaded"
            save("upload-journal.json", journal)
        if not entry.get("block_id"):
            media = {"type": "file_upload", "file_upload": {"id": entry["upload"]["id"]}}
            if kind == "file":
                media["name"] = name
            else:
                media["caption"] = rich(caption)
            result = append(page, [{"object": "block", "type": kind, kind: media}], after)
            entry["block_id"] = result["results"][0]["id"]
            save("upload-journal.json", journal)
        print("Attached", name, flush=True)

    def files():
        page = read("page-properties.json")["id"]
        blocks = children(page)
        table_journal = read("table-journal.json") if (root / "table-journal.json").exists() else {}
        for index, rows in enumerate(read("tables.json"), 1):
            name = str(index)
            if name in table_journal:
                continue
            anchor = next(b["id"] for b in blocks if text_of(b) == f"表：{index}")
            result = append(page, [{"object": "block", "type": "table", "table": {
                "table_width": len(rows[0]), "has_column_header": True, "has_row_header": False,
                "children": [{"object": "block", "type": "table_row",
                              "table_row": {"cells": [rich(cell) for cell in row]}} for row in rows],
            }}], anchor)
            table_journal[name] = result["results"][0]["id"]
            save("table-journal.json", table_journal)
        for name, item in read("images.json").items():
            anchor = next(b["id"] for b in blocks if text_of(b) == "图：" + name)
            attach(page, Path(item["path"]), "image", item["caption"], anchor)
        for name in ("overview.html", "report.md", "edge-analysis-data.zip", "versions.json"):
            attach(page, root / name, "file")

    def links():
        page = read("page-properties.json")
        url, daily_id = page["url"], read("daily-id.json")
        daily = children(daily_id)
        save("daily-blocks-before.json", daily)
        section, tails = None, {}
        for block in daily:
            if block["type"].startswith("heading_"):
                section = text_of(block)
            if section:
                tails[section] = block["id"]
        additions = {
            "今天做了什么": "完成固定NA_out边缘误差新分析，使用固定像素比例、冻结高误差掩膜和无分桶累计曲线。复用112条主结果与24条FT留出；主分析4264项、独立图统计5776项核对通过，未重跑传播或反演。",
            "有哪些发现": "最外约10%有效像素承担FT约71%—76%平方误差；去掉这些评价像素后，剩余RMS下降约44%—48%。边缘集中程度随系统NA_out略增，但边缘绝对RMS不单调，不能单归因于NA。",
            "想法、讨论和新需求": "用户指出径向曲线依赖分桶，提出按高误差像素固定比例选区，并比较系统NA_out。本轮用无噪声FT选择并冻结热点，再评价含噪重复；热点有角向结构，不能把所有高误差像素都视为边缘。",
            "遇到的问题": "冻结热点仍依赖同一主真值，不能替代独立波前泛化验证。配平系统族的遮拦比、基与真值同时变化；当前仅确认边缘误差定位，尚未隔离窗口、平滑、标定或Gibbs机制。",
            "接下来": "后续建议：固定系统和独立真值，对FT窗口、平滑、差分覆盖或标定逐项消融，并先核对网格收敛；本轮没有自动启动这些额外计算。",
            "随手补充": TITLE,
        }
        for section, text in additions.items():
            if any(text in text_of(b) for b in daily):
                continue
            assert section in tails
            append(daily_id, [{"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {
                "rich_text": rich(text, url if section == "随手补充" else None),
            }}], tails[section])
        for original in ("4158b24d-37dc-817f-9ecd-f26116d10207", "4158b24d-37dc-816e-88d1-c51714e8969b"):
            prior = children(original)
            if not any(TITLE in text_of(b) for b in prior):
                append(original, [{"object": "block", "type": "paragraph", "paragraph": {
                    "rich_text": rich("后续分析：" + TITLE, url),
                }}])
        save("links-complete.json", {"report": url, "daily_id": daily_id})

    def verify():
        page = read("page-properties.json")["id"]
        result = run(["ntn", "pages", "get", page, "--json"])
        save("page-readback.json", result)
        assert not result["markdown"]["truncated"]
        assert not result["markdown"]["unknown_block_ids"]
        blocks = children(page)
        save("blocks-final.json", blocks)
        by_id = {b["id"]: b for b in blocks}
        tables = read("tables.json")
        cells_checked = 0
        for index, block_id in read("table-journal.json").items():
            actual_rows = children(block_id)
            actual = [["".join(x.get("plain_text", "") for x in cell)
                       for cell in b["table_row"]["cells"]] for b in actual_rows]
            expected = tables[int(index) - 1]
            assert actual == expected
            cells_checked += sum(map(len, expected))
        downloads = root / "downloaded"
        downloads.mkdir(exist_ok=True)
        receipt = []
        for name, entry in read("upload-journal.json").items():
            block = by_id[entry["block_id"]]
            source = block[entry["kind"]]
            assert source["type"] == "file"
            with urlopen(source["file"]["url"], timeout=120) as response:
                data = response.read()
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            (downloads / name).write_bytes(data)
            receipt.append({"name": name, "sha256": entry["sha256"], "verified": True})
            print("Downloaded and verified", name, flush=True)
        daily = run(["ntn", "pages", "get", read("daily-id.json"), "--json"])
        save("daily-after.json", daily)
        assert page in {r["id"] for r in daily["page"]["properties"]["关联实验"]["relation"]}
        assert result["page"]["properties"]["项目"]["relation"] == [{"id": PROJECT}]
        save("notion-verification.json", {
            "passed": True, "page_id": page, "url": read("page-properties.json")["url"],
            "tables": len(tables), "table_cells_checked": cells_checked,
            "attachments": receipt, "daily_relation_verified": True,
        })

    {"create": create, "files": files, "links": links, "verify": verify}[args.action]()


if __name__ == "__main__":
    main()
