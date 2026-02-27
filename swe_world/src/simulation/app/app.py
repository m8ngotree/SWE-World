import json
import os

from flask import Flask, request, render_template_string, abort
from markupsafe import Markup, escape
import markdown

app = Flask(__name__)

# ===== 固定 JSON 路径，根据你自己的环境修改即可 =====
JSON_PATH = ""
# =====================================================


# 专门渲染 patch diff 的函数
def render_patch(text: str) -> str:
    lines = text.splitlines()
    html_lines = []
    for line in lines:
        cls = ""
        if line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        elif line.startswith("@"):
            cls = "hunk"
        html_lines.append(f"<span class='patch-line {cls}'>{escape(line)}</span>")
    return "<pre class='patch-block'>" + "\n".join(html_lines) + "</pre>"


# 递归渲染函数：根据类型生成对应 HTML
# key: 当前这一层的 key（如果有）
# parent_key: 父级 key，用来继承 patch / 代码文件等语义
def render_value(value, key=None, parent_key=None):
    # 用 key + parent_key 来判断上下文
    context_key = (key or parent_key or "")  # 字符串或空
    context_lower = context_key.lower()

    # dict -> 树形结构（可折叠）
    if isinstance(value, dict):
        items_html = []
        for k, v in value.items():
            child_html = render_value(v, key=k, parent_key=k)
            items_html.append(
                f"""
<li class="tree-item">
  <details open>
    <summary class="tree-key">{escape(str(k))}</summary>
    <div class="tree-children">{child_html}</div>
  </details>
</li>
"""
            )
        ul_html = "<ul class='tree tree-dict'>" + "".join(items_html) + "</ul>"
        return Markup(ul_html)

    # list -> 树形结构（下标作为“文件名”）
    elif isinstance(value, list):
        items_html = []
        # 继承父级的 key（比如 key 里有 patch，就让子元素也按 patch 渲染）
        child_parent_key = key or parent_key
        for idx, item in enumerate(value):
            child_html = render_value(item, key=None, parent_key=child_parent_key)
            items_html.append(
                f"""
<li class="tree-item">
  <details>
    <summary class="tree-key">[{idx}]</summary>
    <div class="tree-children">{child_html}</div>
  </details>
</li>
"""
            )
        ul_html = "<ul class='tree tree-list'>" + "".join(items_html) + "</ul>"
        return Markup(ul_html)

    # 数字 -> 直接显示
    elif isinstance(value, (int, float)):
        return Markup(f"<span class='number'>{value}</span>")

    # 字符串 -> 根据 key 决定是 patch / 代码 / markdown
    elif isinstance(value, str):
        # 1）patch 显示成 diff
        if "patch" in context_lower:
            return Markup(render_patch(value))

        # 2）代码文件（.py / .js / .ts / .java / .json 等）
        _, ext = os.path.splitext(context_key)
        ext = ext.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".json": "json",
            ".sh": "bash",
            ".txt": "text",
        }
        if ext in lang_map:
            lang = lang_map[ext]
            return Markup(
                f"<pre class='code-block'><code class='language-{lang}'>{escape(value)}</code></pre>"
            )

        # 3）普通字符串 -> 当作 Markdown
        md_html = markdown.markdown(value)
        return Markup(f"<div class='markdown-block'>{md_html}</div>")

    # 其他类型 -> 当成字符串/代码显示
    else:
        return Markup(f"<code class='other'>{escape(repr(value))}</code>")


# HTML 模板：树状结构 + 上一条 / 下一条 导航
BASE_TEMPLATE = """
<!doctype html>
<html lang="zh-cn">
<head>
  <meta charset="utf-8">
  <title>JSON 可视化：{{ path }}</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1.5rem;
      background: #f5f5f5;
    }
    h1 {
      margin-top: 0;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
      background: #fff;
      padding: 1.5rem;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .meta {
      font-size: 0.9rem;
      color: #666;
      margin-bottom: 0.5rem;
    }

    /* 记录导航 */
    .record-nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 0.5rem 0 1rem 0;
      padding: 0.5rem 0;
      border-top: 1px solid #eee;
      border-bottom: 1px solid #eee;
    }
    .record-nav-info {
      font-size: 0.95rem;
    }
    .record-buttons {
      display: flex;
      gap: 0.5rem;
    }
    .nav-button {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 999px;
      text-decoration: none;
      border: 1px solid #2563eb;
      color: #2563eb;
      font-size: 0.9rem;
    }
    .nav-button:hover {
      background: #2563eb;
      color: #fff;
    }
    .nav-button.disabled {
      border-color: #ccc;
      color: #aaa;
      cursor: default;
      pointer-events: none;
      background: #f5f5f5;
    }

    /* 树状结构 */
    .tree {
      list-style: none;
      margin: 0.25rem 0 0.25rem 0;
      padding-left: 1rem;
      border-left: 1px dashed #ddd;
    }
    .tree-item {
      margin: 0.1rem 0;
    }
    details {
      padding-left: 0.25rem;
    }
    details > summary {
      list-style: none;
      cursor: pointer;
    }
    details > summary::-webkit-details-marker {
      display: none;
    }
    .tree-key::before {
      content: "📁 ";
      opacity: 0.7;
      font-size: 0.9em;
    }
    .tree-children {
      margin-left: 0.75rem;
      padding-left: 0.5rem;
      border-left: 1px dashed #eee;
    }

    /* 数字 / 代码 / patch */
    .number {
      font-family: "JetBrains Mono", Menlo, Consolas, monospace;
      color: #c7254e;
    }
    pre, code {
      font-family: "JetBrains Mono", Menlo, Consolas, monospace;
      background: #f0f0f0;
      border-radius: 4px;
    }
    pre {
      padding: 0.5rem 0.75rem;
      overflow-x: auto;
      margin: 0.25rem 0 0.5rem 0;
    }
    code {
      padding: 0.15rem 0.25rem;
    }
    .code-block {
      background: #0b1020;
      color: #e5e7eb;
    }

    /* patch diff 样式 */
    .patch-block {
      background: #111827;
      color: #e5e7eb;
    }
    .patch-line {
      display: block;
      white-space: pre;
    }
    .patch-line.add {
      background: #022c22;
      color: #bbf7d0;
    }
    .patch-line.del {
      background: #450a0a;
      color: #fecaca;
    }
    .patch-line.hunk {
      background: #1f2937;
      color: #93c5fd;
    }

    .markdown-block {
      padding: 0.25rem 0.1rem;
    }

    .raw-json-toggle {
      margin-top: 1.5rem;
      border-top: 1px dashed #ddd;
      padding-top: 1rem;
      font-size: 0.9rem;
    }
    details summary {
      cursor: pointer;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>JSON 可视化</h1>
    <div class="meta">
      文件路径：<code>{{ path }}</code>
    </div>

    <div class="record-nav">
      <div class="record-nav-info">
        当前记录：<strong>{{ current_index + 1 }}</strong> / {{ total }}
      </div>
      <div class="record-buttons">
        {% if has_prev %}
          <a href="/?idx={{ prev_index }}" class="nav-button">上一条</a>
        {% else %}
          <span class="nav-button disabled">上一条</span>
        {% endif %}
        {% if has_next %}
          <a href="/?idx={{ next_index }}" class="nav-button">下一条</a>
        {% else %}
          <span class="nav-button disabled">下一条</span>
        {% endif %}
      </div>
    </div>

    <h2>结构化视图（树状，可折叠）</h2>
    <div id="json-view">
      {{ content|safe }}
    </div>

    <div class="raw-json-toggle">
      <details>
        <summary>查看原始 JSON</summary>
        <pre>{{ raw_json }}</pre>
      </details>
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    path = JSON_PATH

    if not os.path.exists(path):
        abort(404, description=f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)[:2]
    except Exception as e:
        abort(500, description=f"Failed to load JSON: {e}")
    print(f"len of data: {len(data)}")
    # 这里假设最外层是 list（多条数据），否则就当成只有一条
    if isinstance(data, list):
        data_list = data
    else:
        data_list = [data]

    total = len(data_list)

    # 通过 URL 参数 ?idx=0/1/2... 来切换记录，只是索引，不改路径
    idx_param = request.args.get("idx", "0")
    try:
        idx = int(idx_param)
    except ValueError:
        idx = 0

    if idx < 0:
        idx = 0
    if idx >= total:
        idx = total - 1

    record = data_list[idx]
    print(f"rendering record {idx}...")
    # 递归转成 HTML（以当前记录为根）
    rendered = render_value(record)

    # 原始 JSON 文本
    raw_json = json.dumps(record, ensure_ascii=False, indent=2)
    print(f"dumping raw json...")

    has_prev = idx > 0
    has_next = idx < total - 1
    prev_index = idx - 1 if has_prev else 0
    next_index = idx + 1 if has_next else idx

    print(f"rendering template...")
    return render_template_string(
        BASE_TEMPLATE,
        content=rendered,
        raw_json=raw_json,
        path=path,
        current_index=idx,
        total=total,
        has_prev=has_prev,
        has_next=has_next,
        prev_index=prev_index,
        next_index=next_index,
    )


if __name__ == "__main__":
    print("starting")
    app.run(debug=True)
