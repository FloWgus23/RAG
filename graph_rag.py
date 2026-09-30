"""
graph_rag.py — Graph mode ของ Flowchat (ค้นข้อมูลจาก "ความสัมพันธ์" ของโค้ด)

ใช้ ast มาตรฐานของ Python วิเคราะห์ไฟล์ .py แล้วสร้างกราฟความสัมพันธ์แบบเบาๆ
(ไม่ต้องพึ่ง dependency หนักอย่าง Kuzu/networkx) เก็บ 3 ความสัมพันธ์หลัก:
    1. โมดูล (ไฟล์)  --import-->      โมดูลอื่นในชุด dataset เดียวกัน
    2. ฟังก์ชัน/เมท็อด --calls-->      ฟังก์ชันอื่น (จับคู่ด้วยชื่อ แบบ best-effort)
    3. โมดูล        --defines-->      ฟังก์ชัน/คลาสที่อยู่ในไฟล์นั้น

ใช้ตอบคำถามเชิงโครงสร้างที่ Vector mode ตอบไม่ได้ เช่น
    "checkout เรียกใช้ฟังก์ชันอะไรบ้าง", "ถ้าแก้ utils.py จะกระทบไฟล์ไหนบ้าง"
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class CodeNode:
    name: str            # ชื่อ unique เช่น "app.main" หรือ "app" (module)
    kind: str            # "module" | "function" | "class" | "method"
    file: str
    lineno: int = 0
    doc: str = ""
    calls: set = field(default_factory=set)         # ชื่อฟังก์ชันที่ node นี้เรียก (เฉพาะ function/method)
    called_by: set = field(default_factory=set)      # ชื่อ node เต็มที่เรียก node นี้
    imports: set = field(default_factory=set)        # เฉพาะ module: ชื่อโมดูลที่ import
    imported_by: set = field(default_factory=set)    # เฉพาะ module: ใครบ้าง import ตัวนี้


class CodeGraph:
    """กราฟความสัมพันธ์ของโค้ด เก็บใน memory ต่อ session เดียวกับ VectorStore"""

    def __init__(self):
        self.nodes: dict[str, CodeNode] = {}

    # -- indexing --------------------------------------------------------
    def clear_file(self, filename: str):
        for key in [k for k, n in self.nodes.items() if n.file == filename]:
            del self.nodes[key]

    def add_file(self, filename: str, source: str, auto_finalize: bool = True) -> int:
        """
        parse ไฟล์ .py หนึ่งไฟล์ เพิ่ม node เข้ากราฟ คืนค่าจำนวน node ที่เพิ่ม

        auto_finalize=False ใช้ตอนอัปโหลดหลายไฟล์พร้อมกัน: เรียก add_file() วนทุกไฟล์
        แบบ auto_finalize=False ก่อน แล้วค่อยเรียก finalize() ครั้งเดียวตอนจบ — ประหยัดกว่า
        การ finalize() ใหม่ทั้งกราฟหลังทุกไฟล์ (ซึ่งจะกลายเป็น O(จำนวนไฟล์ × จำนวน node) โดยไม่จำเป็น)
        และยังถูกต้องกว่าด้วย เพราะ cross-file edges (เช่น A import B) จะครบก็ต่อเมื่อทุกไฟล์ถูกเพิ่มแล้ว
        """
        self.clear_file(filename)
        module_name = re.sub(r"[\\/]", ".", filename.rsplit(".", 1)[0])

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return 0

        module_node = CodeNode(name=module_name, kind="module", file=filename)
        self.nodes[module_name] = module_node

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_node.imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_node.imports.add(node.module.split(".")[0])

        def index_function(fn_node, prefix: str = "", kind: str = "function"):
            full_name = f"{module_name}.{prefix}{fn_node.name}"
            doc = (ast.get_docstring(fn_node) or "").split("\n")[0][:140]
            cnode = CodeNode(name=full_name, kind=kind, file=filename, lineno=fn_node.lineno, doc=doc)
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Call):
                    callee = None
                    if isinstance(sub.func, ast.Name):
                        callee = sub.func.id
                    elif isinstance(sub.func, ast.Attribute):
                        callee = sub.func.attr
                    if callee:
                        cnode.calls.add(callee)
            self.nodes[full_name] = cnode

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                index_function(node)
            elif isinstance(node, ast.ClassDef):
                cls_doc = (ast.get_docstring(node) or "").split("\n")[0][:140]
                self.nodes[f"{module_name}.{node.name}"] = CodeNode(
                    name=f"{module_name}.{node.name}", kind="class", file=filename,
                    lineno=node.lineno, doc=cls_doc,
                )
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        index_function(item, prefix=f"{node.name}.", kind="method")

        if auto_finalize:
            self.finalize()
        return len([n for n in self.nodes.values() if n.file == filename])

    def remove_file(self, filename: str):
        self.clear_file(filename)
        self.finalize()

    def finalize(self):
        """คำนวณความสัมพันธ์ย้อนกลับใหม่ทั้งหมด (called_by / imported_by) แบบ best-effort จับคู่ชื่อ"""
        for n in self.nodes.values():
            n.called_by.clear()
            n.imported_by.clear()

        by_short_name: dict[str, list[str]] = {}
        for key, n in self.nodes.items():
            if n.kind in ("function", "method"):
                by_short_name.setdefault(key.rsplit(".", 1)[-1], []).append(key)

        for key, n in self.nodes.items():
            if n.kind in ("function", "method"):
                for callee in n.calls:
                    for target in by_short_name.get(callee, []):
                        if target != key:
                            self.nodes[target].called_by.add(key)
            elif n.kind == "module":
                for imp in n.imports:
                    if imp in self.nodes:
                        self.nodes[imp].imported_by.add(n.name)

    # -- query -------------------------------------------------------------
    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def file_names(self) -> list[str]:
        seen = []
        for n in self.nodes.values():
            if n.file not in seen:
                seen.append(n.file)
        return seen

    def node_count(self, filename: str | None = None) -> int:
        if filename is None:
            return len(self.nodes)
        return sum(1 for n in self.nodes.values() if n.file == filename)

    def find_matches(self, query: str, limit: int = 5) -> list[CodeNode]:
        """หา node (ฟังก์ชัน/คลาส/โมดูล) ที่ชื่อปรากฏอยู่ในคำถามของผู้ใช้"""
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query))
        matches = [n for key, n in self.nodes.items() if key.rsplit(".", 1)[-1] in tokens]
        matches.sort(key=lambda n: 0 if n.kind in ("function", "method") else 1)
        return matches[:limit]

    def describe(self, node: CodeNode) -> str:
        lines = [f"[{node.kind}] {node.name}  (ไฟล์: {node.file}, บรรทัด {node.lineno})"]
        if node.doc:
            lines.append(f"คำอธิบาย: {node.doc}")
        if node.kind in ("function", "method"):
            if node.calls:
                lines.append(f"เรียกใช้: {', '.join(sorted(node.calls))}")
            if node.called_by:
                callers = ", ".join(k.rsplit(".", 1)[-1] for k in sorted(node.called_by))
                lines.append(f"ถูกเรียกโดย: {callers}")
        elif node.kind == "module":
            if node.imports:
                lines.append(f"import: {', '.join(sorted(node.imports))}")
            if node.imported_by:
                lines.append(f"ถูก import โดย: {', '.join(sorted(node.imported_by))}")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 5):
        """คืนผลลัพธ์รูปแบบเดียวกับ VectorStore.search() เพื่อให้ app.py ใช้โค้ดร่วมกันได้"""
        matches = self.find_matches(query, limit=top_k)
        return [{"text": self.describe(n), "source": n.file, "node": n.name} for n in matches]
