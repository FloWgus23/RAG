"""
graph_rag.py — Graph mode ของ Flowchat (ค้นข้อมูลจาก "ความสัมพันธ์" ของโค้ด)

ใช้ ast มาตรฐานของ Python วิเคราะห์ไฟล์ .py แล้วสร้างกราฟความสัมพันธ์แบบเบาๆ
(ไม่ต้องพึ่ง dependency หนักอย่าง Kuzu/networkx) เก็บความสัมพันธ์หลัก:
    1. โมดูล (ไฟล์)   --import-->    โมดูลอื่นในชุด dataset เดียวกัน
    2. ฟังก์ชัน/เมท็อด --calls-->     ฟังก์ชันอื่น (รวมโค้ดระดับโมดูลที่เรียกฟังก์ชันตอนรันไฟล์)
    3. โมดูล         --defines-->   ฟังก์ชัน/คลาสที่อยู่ในไฟล์นั้น
    4. คลาส          --extends-->   คลาสแม่ (ใช้หาเมท็อดที่สืบทอดมา)

อัปเกรดสำคัญ
- แก้ปัญหาชื่อชนกัน (name collision): เดิมจับคู่การเรียกด้วย "ชื่อ" ล้วนๆ ทำให้ log() ในหลายไฟล์
  โยงมั่ว ตอนนี้ใช้ข้อมูล import (import x / from x import y as z / import x as m) ระบุว่า
  ชื่อที่ถูกเรียกมาจากไฟล์ไหนจริง และถ้าชื่อนั้นมาจากไลบรารีภายนอก (เช่น math.log) จะไม่โยงเข้า dataset
  สำหรับ obj.method() ใช้ type hint / การสร้างอ็อบเจ็กต์ (x = Foo()) / self ช่วยหาคลาสของ obj
  ถ้ายังไม่รู้ชนิดจริงๆ จะเก็บเป็นความสัมพันธ์ "ไม่แน่ใจ" (แสดงเป็น (?)) แยกจากที่แน่ใจ
- Multi-hop traversal: ไล่ความสัมพันธ์ได้หลายชั้น เช่น "ถ้าแก้ A จะกระทบอะไรบ้าง" ไล่ย้อนตามสายการเรียก
  เป็นทอดๆ (A ← B ← C ...) ได้สูงสุด depth ชั้น พร้อมสรุปจำนวน node/ไฟล์ที่ได้รับผลกระทบ
  และไล่ ไฟล์ที่ import ต่อกันเป็นทอดๆ สำหรับ node ประเภทโมดูล

ใช้ตอบคำถามเชิงโครงสร้างที่ Vector mode ตอบไม่ได้ เช่น
    "checkout เรียกใช้ฟังก์ชันอะไรบ้าง", "ถ้าแก้ utils.py จะกระทบไฟล์ไหนบ้าง"
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# ชื่อเมท็อดยอดนิยมของ builtin/container (list.append, dict.get, str.join ฯลฯ) — ถ้าไม่รู้ชนิดของ obj
# จะไม่เดาโยงชื่อเหล่านี้เข้ากับเมท็อดใน dataset เพราะมักเป็นการเรียก builtin มากกว่า
_COMMON_METHODS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "copy", "index", "count", "sort", "reverse",
    "get", "set", "add", "update", "items", "keys", "values", "setdefault", "discard", "union",
    "join", "split", "strip", "lstrip", "rstrip", "replace", "format", "startswith", "endswith",
    "lower", "upper", "encode", "decode", "find", "read", "write", "close", "open", "seek",
})
_AMBIGUOUS_LIMIT = 5   # ถ้าชื่อเมท็อดเดียวกันมีเกินนี้ใน dataset ถือว่ากำกวมเกินไป ไม่เดา
_LIST_CAP = 12         # จำกัดจำนวนรายการต่อชั้นตอนแสดงผล ไม่ให้ context ที่ส่งให้ LLM บวมเกินไป


@dataclass
class CodeNode:
    name: str            # ชื่อ unique เช่น "app.main" หรือ "app" (module)
    kind: str            # "module" | "function" | "class" | "method"
    file: str
    lineno: int = 0
    doc: str = ""
    module: str = ""                                 # ชื่อโมดูลเจ้าของ node นี้ (module เองชี้มาที่ตัวเอง)
    owner_class: str = ""                            # เฉพาะ method: ชื่อเต็มของคลาสเจ้าของ เช่น "app.Foo"
    calls: set = field(default_factory=set)          # ชื่อ (ดิบ) ที่ node นี้เรียก — ใช้แสดง/ตรวจสอบ
    call_refs: set = field(default_factory=set)      # การเรียกแบบมีโครงสร้าง: ("name", f, None) | ("attr", base, attr)
    resolved_calls: set = field(default_factory=set)     # ชื่อเต็มของ node ที่เรียกแน่ๆ (ยืนยันจาก import/type)
    ambiguous_calls: set = field(default_factory=set)    # เรียกจริงหรือไม่ไม่แน่ใจ (จับคู่จากชื่อเมท็อดอย่างเดียว)
    called_by: set = field(default_factory=set)      # ชื่อ node เต็มที่เรียก node นี้ (แน่ใจ)
    maybe_called_by: set = field(default_factory=set)    # ชื่อ node เต็มที่อาจเรียก node นี้ (ไม่แน่ใจ)
    imports: set = field(default_factory=set)        # เฉพาะ module: ชื่อโมดูลที่ import
    imported_by: set = field(default_factory=set)    # เฉพาะ module: ใครบ้าง import ตัวนี้
    # -- ข้อมูลสำหรับ resolve ชื่อ (ใช้ตอน finalize) --
    import_aliases: dict = field(default_factory=dict)   # module: ชื่อที่ใช้ในไฟล์ -> ชื่อโมดูลเต็ม
    from_imports: dict = field(default_factory=dict)     # module: ชื่อที่ใช้ในไฟล์ -> (โมดูล, ชื่อเดิม)
    star_imports: list = field(default_factory=list)     # module: from x import *
    rel_imports: list = field(default_factory=list)      # module: from .x import y -> (level, โมดูลส่วนหลังจุด, [(ชื่อที่ใช้, ชื่อเดิม)])
    rel_added: dict = field(default_factory=dict)        # module: สิ่งที่ _apply_relative_imports เพิ่มไว้ (ไว้ถอนก่อนคำนวณใหม่)
    var_types: dict = field(default_factory=dict)        # module/function: ตัวแปร -> ชื่อชนิด (จาก hint / Foo())
    bases: list = field(default_factory=list)            # class: ชื่อคลาสแม่ตามที่เขียนในโค้ด
    resolved_bases: list = field(default_factory=list)   # class: ชื่อเต็มของคลาสแม่ที่อยู่ใน dataset


# ---------------------------------------------------------------------------
# ฟังก์ชันช่วยอ่าน AST
# ---------------------------------------------------------------------------
def _dotted(expr) -> str | None:
    """a.b.c -> "a.b.c" (เฉพาะกรณีที่ทั้งสายเป็น Name/Attribute ล้วนๆ) ไม่ใช่ -> None"""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def _ann_name(ann) -> str | None:
    """ดึงชื่อชนิดจาก type annotation: Foo, mod.Foo, "Foo", Optional[Foo], Foo | None"""
    if ann is None:
        return None
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        s = ann.value.strip()
        return s if re.fullmatch(r"[A-Za-z_][\w.]*", s) else None
    if isinstance(ann, (ast.Name, ast.Attribute)):
        return _dotted(ann)
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        for side in (ann.left, ann.right):
            if isinstance(side, ast.Constant) and side.value is None:
                continue
            name = _ann_name(side)
            if name and name != "None":
                return name
        return None
    if isinstance(ann, ast.Subscript) and _dotted(ann.value) in ("Optional", "typing.Optional"):
        return _ann_name(ann.slice)
    return None


def _walk_scope(stmts):
    """เดินโหนดใน scope นี้ โดยไม่เข้าไปในตัว def/class (โค้ดข้างในเป็นของ node อื่น)"""
    stack = list(reversed(stmts))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(n))))


def _collect_calls(nodes) -> set:
    refs = set()
    for sub in nodes:
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                refs.add(("name", f.id, None))
            elif isinstance(f, ast.Attribute):
                refs.add(("attr", _dotted(f.value), f.attr))
    return refs


def _collect_var_types(nodes, into: dict):
    """เก็บชนิดของตัวแปรจาก `x: Foo = ...` และ `x = Foo(...)` (annotation ชนะการเดาจาก constructor)"""
    for sub in nodes:
        if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            t = _ann_name(sub.annotation)
            if t:
                into[sub.target.id] = t
        elif isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
            t = _dotted(sub.value.func)
            if t:
                for tg in sub.targets:
                    if isinstance(tg, ast.Name):
                        into.setdefault(tg.id, t)


def _param_types(fn) -> dict:
    types = {}
    a = fn.args
    for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
        t = _ann_name(arg.annotation)
        if t:
            types[arg.arg] = t
    return types


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

        mod = CodeNode(name=module_name, kind="module", file=filename, module=module_name,
                       doc=(ast.get_docstring(tree) or "").split("\n")[0][:140])
        self.nodes[module_name] = mod

        # -- import ทั้งหมดในไฟล์ (รวม import ที่อยู่ในฟังก์ชัน) --
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod.imports.add(alias.name)
                    if alias.asname:
                        mod.import_aliases[alias.asname] = alias.name
                    else:   # `import a.b` ผูกชื่อ "a" ในไฟล์
                        top = alias.name.split(".")[0]
                        mod.import_aliases[top] = top
            elif isinstance(node, ast.ImportFrom):
                if node.level:   # from .x import y / from .. import z -> เก็บดิบไว้ แปลงเป็นชื่อเต็มตอน finalize (ต้องรู้ว่าไฟล์นี้เป็นแพ็กเกจหรือไม่)
                    mod.rel_imports.append(
                        (node.level, node.module or "", [(a.asname or a.name, a.name) for a in node.names])
                    )
                    continue
                base = node.module or ""
                if base:
                    mod.imports.add(base)
                for alias in node.names:
                    if alias.name == "*":
                        if base:
                            mod.star_imports.append(base)
                        continue
                    mod.from_imports[alias.asname or alias.name] = (base, alias.name)
                    if not base:            # from . import x  -> x เป็นโมดูล
                        mod.imports.add(alias.name)

        # -- โค้ดระดับโมดูล (สคริปต์ที่รันตอนเปิดไฟล์ เช่น app.py ของ Streamlit) --
        scope = list(_walk_scope(tree.body))
        mod.call_refs = _collect_calls(scope)
        mod.calls = {r[1] if r[0] == "name" else r[2] for r in mod.call_refs}
        _collect_var_types(scope, mod.var_types)

        def index_function(fn_node, prefix: str = "", kind: str = "function", owner: str = ""):
            full_name = f"{module_name}.{prefix}{fn_node.name}"
            doc = (ast.get_docstring(fn_node) or "").split("\n")[0][:140]
            cnode = CodeNode(name=full_name, kind=kind, file=filename, lineno=fn_node.lineno, doc=doc,
                             module=module_name, owner_class=owner)
            body_nodes = list(ast.walk(fn_node))
            cnode.call_refs = _collect_calls(body_nodes)
            cnode.calls = {r[1] if r[0] == "name" else r[2] for r in cnode.call_refs}
            cnode.var_types = _param_types(fn_node)
            _collect_var_types(body_nodes, cnode.var_types)
            self.nodes[full_name] = cnode

        func_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        for node in tree.body:
            if isinstance(node, func_types):
                index_function(node)
            elif isinstance(node, ast.ClassDef):
                cls_key = f"{module_name}.{node.name}"
                cls_doc = (ast.get_docstring(node) or "").split("\n")[0][:140]
                bases = [b for b in (_dotted(x) for x in node.bases) if b]
                self.nodes[cls_key] = CodeNode(
                    name=cls_key, kind="class", file=filename, lineno=node.lineno, doc=cls_doc,
                    module=module_name, bases=bases,
                )
                for item in node.body:
                    if isinstance(item, func_types):
                        index_function(item, prefix=f"{node.name}.", kind="method", owner=cls_key)

        if auto_finalize:
            self.finalize()
        return len([n for n in self.nodes.values() if n.file == filename])

    def remove_file(self, filename: str):
        self.clear_file(filename)
        self.finalize()

    # -- resolve ชื่อ (หัวใจของการแก้ name collision) -----------------------
    def _find_module(self, dotted: str) -> str | None:
        """หาโมดูลใน dataset ที่ตรงกับชื่อ (ตรงเป๊ะ หรือเป็นส่วนท้ายของ path เช่น pkg.utils)"""
        if not dotted:
            return None
        n = self.nodes.get(dotted)
        if n and n.kind == "module":
            return dotted
        for key, node in self.nodes.items():
            if node.kind == "module" and key.endswith("." + dotted):
                return key
        return None

    def _resolve_symbol(self, mod: str, name: str, _seen: frozenset = frozenset()) -> str | None:
        """
        ชื่อ `name` ที่เขียนในไฟล์ `mod` ชี้ไปที่ node ไหนใน dataset (function/class/module)
        ลำดับ: นิยามในไฟล์เอง -> from x import name -> import x as name -> from x import *
        คืน None ถ้าเป็น builtin / ไลบรารีภายนอก / หาไม่เจอ
        """
        key = f"{mod}.{name}"
        if key in self.nodes:
            return key
        m = self.nodes.get(mod)
        if m is None or (mod, name) in _seen:
            return None
        seen = _seen | {(mod, name)}
        if name in m.from_imports:
            src, orig = m.from_imports[name]
            src_mod = self._find_module(src)
            if src_mod:
                hit = self._resolve_symbol(src_mod, orig, seen)   # ตามรอย re-export ได้ต่อ
                if hit:
                    return hit
            return self._find_module(f"{src}.{orig}" if src else orig)   # from pkg import submodule
        if name in m.import_aliases:
            return self._find_module(m.import_aliases[name])
        for star in m.star_imports:
            src_mod = self._find_module(star)
            if src_mod:
                hit = self._resolve_symbol(src_mod, name, seen)
                if hit:
                    return hit
        return None

    def _is_external_name(self, mod: str, name: str) -> bool:
        """ชื่อนี้ถูก import เข้ามา แต่ไม่ได้ชี้ไปที่ไฟล์ใน dataset = มาจากไลบรารีภายนอก (เช่น streamlit)"""
        m = self.nodes.get(mod)
        if m is None:
            return False
        imported = name in m.from_imports or name in m.import_aliases
        return imported and self._resolve_symbol(mod, name) is None

    def _resolve_module_expr(self, mod: str, dotted: str) -> str | None:
        """นิพจน์ a.b.c ที่หมายถึง "โมดูล" ใน dataset (ผ่าน import) -> ชื่อโมดูล ไม่ใช่ -> None"""
        first, *rest = dotted.split(".")
        m = self.nodes.get(mod)
        if m is None:
            return None
        if first in m.import_aliases:
            return self._find_module(".".join([m.import_aliases[first], *rest]))
        sym = self._resolve_symbol(mod, first)
        if sym and self.nodes[sym].kind == "module":
            return self._find_module(".".join([sym, *rest])) if rest else sym
        return None

    def _resolve_class_expr(self, mod: str, expr: str) -> str | None:
        if "." not in expr:
            key = self._resolve_symbol(mod, expr)
        else:
            head, last = expr.rsplit(".", 1)
            src = self._resolve_module_expr(mod, head)
            key = f"{src}.{last}" if src else None
        return key if key in self.nodes and self.nodes[key].kind == "class" else None

    def _classify_type(self, mod: str, type_name: str):
        """("class", key) = คลาสใน dataset | ("external", None) = ไลบรารีภายนอก | ("unknown", None)"""
        cls = self._resolve_class_expr(mod, type_name)
        if cls:
            return "class", cls
        if self._is_external_name(mod, type_name.split(".")[0]):
            return "external", None
        return "unknown", None

    def _lookup_method(self, cls_key: str, name: str, _seen: frozenset = frozenset()) -> str | None:
        """หาเมท็อดในคลาส ถ้าไม่มีให้ไล่ขึ้นไปหาในคลาสแม่ (รองรับ inheritance)"""
        if cls_key in _seen:
            return None
        key = f"{cls_key}.{name}"
        if key in self.nodes and self.nodes[key].kind in ("method", "function"):
            return key
        for base in self.nodes[cls_key].resolved_bases:
            hit = self._lookup_method(base, name, _seen | {cls_key})
            if hit:
                return hit
        return None

    def _call_targets(self, key: str) -> set:
        """เรียก class = สร้างอ็อบเจ็กต์ -> โยงไป __init__ (ถ้าไม่มี __init__ โยงไปที่ตัวคลาสเอง)"""
        n = self.nodes[key]
        if n.kind == "class":
            init = f"{key}.__init__"
            return {init} if init in self.nodes else {key}
        if n.kind in ("function", "method"):
            return {key}
        return set()

    def _ambiguous_methods(self, attr: str, caller_key: str) -> set:
        """ไม่รู้ชนิดของ obj: เดาจากชื่อเมท็อดอย่างเดียว (ผลลัพธ์ถูกติดป้าย "ไม่แน่ใจ")"""
        if attr in _COMMON_METHODS:
            return set()
        cands = {k for k, n in self.nodes.items()
                 if n.kind == "method" and k.rsplit(".", 1)[-1] == attr and k != caller_key}
        return cands if len(cands) <= _AMBIGUOUS_LIMIT else set()

    def _resolve_call(self, caller_key: str, caller: CodeNode, ref: tuple) -> tuple[set, set]:
        """คืน (เรียกแน่ๆ, ไม่แน่ใจ) ของการเรียกหนึ่งครั้ง"""
        mod = caller.module
        kind, a, b = ref

        if kind == "name":                                   # foo()
            sym = self._resolve_symbol(mod, a)
            return (self._call_targets(sym), set()) if sym else (set(), set())

        base, attr = a, b                                    # base.attr()
        if base is None:
            return set(), self._ambiguous_methods(attr, caller_key)

        parts = base.split(".")
        first = parts[0]
        mod_node = self.nodes[mod]

        # 1) self.method() / cls.method() — หาในคลาสตัวเอง (รวมคลาสแม่)
        if len(parts) == 1 and first in ("self", "cls") and caller.owner_class:
            hit = self._lookup_method(caller.owner_class, attr)
            return ({hit} if hit else set()), set()

        # 2) ตัวแปรที่รู้ชนิด (type hint / x = Foo())
        tname = caller.var_types.get(first) or mod_node.var_types.get(first)
        if tname and first not in ("self", "cls"):
            what, cls = self._classify_type(mod, tname)
            if what == "class":
                if len(parts) == 1:
                    hit = self._lookup_method(cls, attr)
                    return ({hit} if hit else set()), set()
                return set(), self._ambiguous_methods(attr, caller_key)   # x.field.method() ไม่รู้ชนิดของ field
            if what == "external":
                return set(), set()

        # 3) ชื่อที่ import มา: โมดูล.ฟังก์ชัน() หรือ คลาส.เมท็อด()
        if len(parts) == 1:
            sym = self._resolve_symbol(mod, first)
            if sym:
                sk = self.nodes[sym].kind
                if sk == "module":
                    tgt = f"{sym}.{attr}"
                    return (self._call_targets(tgt), set()) if tgt in self.nodes else (set(), set())
                if sk == "class":
                    hit = self._lookup_method(sym, attr)
                    return ({hit} if hit else set()), set()
                return set(), set()
        else:
            src = self._resolve_module_expr(mod, base)             # pkg.mod.func()
            if src:
                tgt = f"{src}.{attr}"
                return (self._call_targets(tgt), set()) if tgt in self.nodes else (set(), set())
            head, last = base.rsplit(".", 1)                      # mod.Class.method()
            src = self._resolve_module_expr(mod, head)
            if src and f"{src}.{last}" in self.nodes and self.nodes[f"{src}.{last}"].kind == "class":
                hit = self._lookup_method(f"{src}.{last}", attr)
                return ({hit} if hit else set()), set()

        # 4) ต้นทางเป็นไลบรารีภายนอก (st.xxx, ollama.xxx) -> ไม่โยงเข้า dataset
        if self._is_external_name(mod, first):
            return set(), set()

        # 5) ไม่รู้ชนิดของ obj จริงๆ -> เดาจากชื่อเมท็อด แต่ติดป้าย "ไม่แน่ใจ"
        return set(), self._ambiguous_methods(attr, caller_key)

    def _apply_relative_imports(self):
        """
        แปลง relative import (from .x import y / from ..pkg import z) เป็นชื่อโมดูลเต็ม
        ชื่อไฟล์ใน dataset ถูก "แบน" (smolml/utils/__init__.py -> smolml.utils.py) จึงต้องเดาว่าไฟล์ไหนเป็นแพ็กเกจ:
        ถ้ามีโมดูลอื่นชื่อขึ้นต้นด้วย "<ชื่อนี้>." แสดงว่าเป็นแพ็กเกจ (__init__) ไม่งั้นเป็นโมดูลธรรมดา
        แพ็กเกจปัจจุบัน = ตัวมันเองถ้าเป็นแพ็กเกจ, ไม่งั้นคือโมดูลแม่ แล้วขึ้นไปอีก (level-1) ชั้น
        """
        mods = [k for k, n in self.nodes.items() if n.kind == "module"]
        for key in mods:
            n = self.nodes[key]
            prev = n.rel_added                                  # ถอนของรอบก่อนออกก่อน (ผลลัพธ์อาจเปลี่ยนเมื่อมีไฟล์ใหม่เข้ามา)
            for x in prev.get("imports", ()):
                n.imports.discard(x)
            for name in prev.get("from", ()):
                n.from_imports.pop(name, None)
            for x in prev.get("star", ()):
                if x in n.star_imports:
                    n.star_imports.remove(x)
            added = {"imports": set(), "from": set(), "star": []}
            n.rel_added = added
            if not n.rel_imports:
                continue

            prefix = key + "."
            is_pkg = any(m.startswith(prefix) for m in mods)
            parts = key.split(".")
            pkg_parts = parts if is_pkg else parts[:-1]

            for level, base, names in n.rel_imports:
                up = level - 1
                if up > len(pkg_parts):
                    continue                                    # อ้างขึ้นเกินราก ข้าม
                pkg = ".".join(pkg_parts[: len(pkg_parts) - up])
                base_abs = ".".join(p for p in (pkg, base) if p)
                if base and base_abs and base_abs not in n.imports:
                    n.imports.add(base_abs)
                    added["imports"].add(base_abs)
                for local, orig in names:
                    if orig == "*":
                        if base_abs:
                            n.star_imports.append(base_abs)
                            added["star"].append(base_abs)
                        continue
                    if local not in n.from_imports:
                        n.from_imports[local] = (base_abs, orig)
                        added["from"].add(local)
                    if not base_abs:                            # from . import x ที่ระดับบนสุด -> x เป็นโมดูล
                        n.imports.add(orig)
                        added["imports"].add(orig)

    def finalize(self):
        """คำนวณความสัมพันธ์ทั้งหมดใหม่ (calls / called_by / imported_by) โดย resolve ชื่อผ่าน import และชนิดของ obj"""
        self._apply_relative_imports()
        for n in self.nodes.values():
            n.called_by.clear()
            n.maybe_called_by.clear()
            n.imported_by.clear()
            n.resolved_calls.clear()
            n.ambiguous_calls.clear()
            n.resolved_bases.clear()

        for key, n in self.nodes.items():                      # คลาสแม่ (ต้องรู้ก่อนหาเมท็อดที่สืบทอด)
            if n.kind == "class":
                for b in n.bases:
                    base_key = self._resolve_class_expr(n.module, b)
                    if base_key and base_key != key:
                        n.resolved_bases.append(base_key)

        for key, n in self.nodes.items():                      # การเรียกใช้
            amb_all: set = set()
            for ref in n.call_refs:
                certain, amb = self._resolve_call(key, n, ref)
                n.resolved_calls |= certain
                amb_all |= amb
            n.resolved_calls.discard(key)                      # recursion ไม่นับเป็นความสัมพันธ์
            n.ambiguous_calls = amb_all - n.resolved_calls - {key}
            for t in n.resolved_calls:
                self.nodes[t].called_by.add(key)
            for t in n.ambiguous_calls:
                self.nodes[t].maybe_called_by.add(key)

        for n in self.nodes.values():                          # import ระหว่างไฟล์
            if n.kind != "module":
                continue
            deps = set(n.imports)
            for src, orig in n.from_imports.values():
                deps.add(f"{src}.{orig}" if src else orig)     # from pkg import submodule
            for d in deps:
                target = self._find_module(d)
                if target and target != n.name:
                    self.nodes[target].imported_by.add(n.name)

    # -- multi-hop traversal -------------------------------------------------
    def _start_nodes(self, key: str) -> list[str]:
        """คลาส = ตัวคลาส + เมท็อดทั้งหมดของมัน (ถูกกระทบเมื่อมีใครเรียกอะไรก็ตามในคลาส)"""
        n = self.nodes[key]
        if n.kind == "class":
            return [key] + [k for k, m in self.nodes.items() if m.owner_class == key]
        return [key]

    def traverse(self, key: str, direction: str = "callers", depth: int = 3) -> list[list[tuple[str, bool]]]:
        """
        ไล่กราฟทีละชั้น (BFS) จาก node `key` สูงสุด `depth` ชั้น กัน cycle ด้วย visited
        direction: "callers" = ไล่ย้อนหาคนเรียก (ถ้าแก้ key จะกระทบใคร)
                   "callees" = ไล่ไปข้างหน้าหาสิ่งที่ถูกเรียกต่อเป็นทอดๆ
        คืน [[(node, uncertain), ...] ชั้น 1, ชั้น 2, ...] — uncertain=True ถ้าเส้นทางต้องผ่านความสัมพันธ์ "ไม่แน่ใจ"
        """
        starts = self._start_nodes(key)
        visited = set(starts)
        frontier = {s: False for s in starts}
        levels: list[list[tuple[str, bool]]] = []
        for _ in range(max(depth, 0)):
            nxt: dict[str, bool] = {}
            for cur, cur_unc in frontier.items():
                node = self.nodes[cur]
                sure, maybe = ((node.called_by, node.maybe_called_by) if direction == "callers"
                               else (node.resolved_calls, node.ambiguous_calls))
                for t, unc in [(t, cur_unc) for t in sure] + [(t, True) for t in maybe]:
                    if t in visited or t not in self.nodes:
                        continue
                    nxt[t] = (nxt[t] and unc) if t in nxt else unc
            if not nxt:
                break
            visited.update(nxt)
            levels.append(sorted(nxt.items()))
            frontier = nxt
        return levels

    def import_levels(self, module_key: str, depth: int = 3) -> list[list[str]]:
        """ไล่ "ไฟล์ที่ import ต่อกันเป็นทอดๆ" ย้อนกลับ: ใครใช้โมดูลนี้ ใครใช้ตัวที่ใช้ ... (สูงสุด depth ชั้น)"""
        visited = {module_key}
        frontier = [module_key]
        levels = []
        for _ in range(max(depth, 0)):
            nxt = sorted({m for cur in frontier for m in self.nodes[cur].imported_by} - visited)
            if not nxt:
                break
            visited.update(nxt)
            levels.append(nxt)
            frontier = nxt
        return levels

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
        """
        หา node (ฟังก์ชัน/คลาส/โมดูล) ที่ชื่อปรากฏอยู่ในคำถามของผู้ใช้
        ถ้าผู้ใช้พิมพ์ชื่อเต็ม เช่น VectorStore.search หรือ app.main จะจับคู่ตรงตัวก่อน (แก้ชื่อซ้ำข้ามคลาส/ไฟล์)
        """
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query))
        dotted = set(re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", query))

        def exact(key: str) -> bool:
            return any(key == d or key.endswith("." + d) for d in dotted)

        exact_hits = [n for key, n in self.nodes.items() if exact(key)]
        if exact_hits:    # ผู้ใช้ระบุชื่อเต็มชัดเจน -> ตอบเฉพาะตัวนั้น ไม่พ่วงตัวชื่อสั้นเหมือนกันมาปน
            return exact_hits[:limit]
        matches = [n for key, n in self.nodes.items() if key.rsplit(".", 1)[-1] in tokens]
        matches.sort(key=lambda n: 0 if n.kind in ("function", "method") else 1)
        return matches[:limit]

    @staticmethod
    def _fmt_items(items: list[tuple[str, bool]], cap: int = _LIST_CAP) -> str:
        shown = [f"{k}{' (?)' if unc else ''}" for k, unc in items[:cap]]
        if len(items) > cap:
            shown.append(f"... (+{len(items) - cap} รายการ)")
        return ", ".join(shown)

    def describe(self, node: CodeNode, depth: int = 3) -> str:
        lines = [f"[{node.kind}] {node.name}  (ไฟล์: {node.file}, บรรทัด {node.lineno})"]
        if node.doc:
            lines.append(f"คำอธิบาย: {node.doc}")

        if node.kind == "module":
            defined = [k for k, n in self.nodes.items() if n.module == node.name and n.kind != "module"]
            if defined:
                lines.append("นิยาม: " + ", ".join(k[len(node.name) + 1:] for k in defined[:_LIST_CAP])
                             + (f" ... (+{len(defined) - _LIST_CAP})" if len(defined) > _LIST_CAP else ""))
            if node.imports:
                lines.append(f"(A) โมดูลนี้ import อะไรบ้าง (this imports): {', '.join(sorted(node.imports))}")
            if node.resolved_calls:
                lines.append("(A ต่อ) โค้ดระดับโมดูล (ตอนรันไฟล์) เรียกใช้: "
                             + self._fmt_items([(k, False) for k in sorted(node.resolved_calls)]))
            levels = self.import_levels(node.name, depth)
            if levels:
                lines.append(f"(B) ใครบ้าง import โมดูลนี้ / ผลกระทบถ้าแก้ไฟล์นี้ (others import this / impact if changed), ไล่สูงสุด {depth} ชั้น:")
                for i, lv in enumerate(levels, 1):
                    lines.append(f"  ชั้น {i}{' (import ตรงๆ)' if i == 1 else ''}: {', '.join(lv)}")
                total = sorted({m for lv in levels for m in lv})
                lines.append(f"  → กระทบรวม {len(total)} ไฟล์: {', '.join(total)}")
            else:
                lines.append("(B) ไม่พบไฟล์อื่นใน dataset ที่ import โมดูลนี้ (ไม่มีใครถูกกระทบถ้าแก้ไฟล์นี้)")
            return "\n".join(lines)

        if node.kind == "class":
            if node.bases:
                resolved = {k.rsplit(".", 1)[-1]: k for k in node.resolved_bases}
                shown = [resolved.get(b.rsplit(".", 1)[-1], b) for b in node.bases]
                lines.append(f"สืบทอดจาก: {', '.join(shown)}")
            methods = [k[len(node.name) + 1:] for k, m in self.nodes.items() if m.owner_class == node.name]
            if methods:
                lines.append(f"เมท็อด: {', '.join(methods[:_LIST_CAP])}")
        else:
            direct = ([(k, False) for k in sorted(node.resolved_calls)]
                      + [(k, True) for k in sorted(node.ambiguous_calls)])
            if direct:
                lines.append(f"(A) node นี้เรียกใช้อะไรบ้าง / this CALLS OUT TO: {self._fmt_items(direct)}")
            else:
                lines.append("(A) node นี้เรียกใช้อะไรบ้าง / this CALLS OUT TO: ไม่มี ไม่เรียกฟังก์ชันอื่นใน dataset นี้เลย")
            deeper = self.traverse(node.name, "callees", depth)[1:]   # ชั้น 1 แสดงไปแล้วในบรรทัด (A)
            if deeper:
                lines.append(f"(A ต่อ) เรียกต่อเป็นทอดๆ (ไล่ลงไปอีก สูงสุด {depth} ชั้น):")
                for i, lv in enumerate(deeper, 2):
                    lines.append(f"  ชั้น {i}: {self._fmt_items(lv)}")

        up = self.traverse(node.name, "callers", depth)
        if up:
            lines.append(f"(B) ใครเรียกใช้ node นี้บ้าง / ผลกระทบถ้าแก้ตัวนี้ (WHO CALLS THIS / impact if changed), ไล่สูงสุด {depth} ชั้น:")
            for i, lv in enumerate(up, 1):
                lines.append(f"  ชั้น {i}{' (เรียกตรงๆ)' if i == 1 else ''}: {self._fmt_items(lv)}")
            affected = {k for lv in up for k, _ in lv}
            files = sorted({self.nodes[k].file for k in affected})
            lines.append(f"  → กระทบรวม {len(affected)} node ใน {len(files)} ไฟล์: {', '.join(files)}")
            if any(unc for lv in up for _, unc in lv):
                lines.append("  (?) = ไม่แน่ใจ: จับคู่จากชื่อเมท็อดอย่างเดียว ยังไม่ทราบชนิดของอ็อบเจ็กต์ที่เรียก")
        else:
            lines.append("(B) ใครเรียกใช้ node นี้บ้าง / WHO CALLS THIS: ไม่มีใครเรียกจากที่อื่นใน dataset นี้เลย "
                          "(อาจเป็นจุดเริ่มต้นของโปรแกรม, callback, หรือโค้ดที่ไม่ได้ใช้)")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 5, depth: int | None = None):
        """
        คืนผลลัพธ์รูปแบบเดียวกับ VectorStore.search() เพื่อให้ app.py ใช้โค้ดร่วมกันได้

        depth=None (ค่าเริ่มต้น): เลือก depth อัตโนมัติตามคำถาม — ถ้าคำถามมีคำว่า "กระทบ"/"impact"/
        "ถ้าแก้"/"ผลกระทบ" จะไล่ multi-hop เต็ม (3 ชั้น) เพื่อวิเคราะห์ผลกระทบเชิงลึก แต่ถ้าเป็นคำถามทั่วไป
        (เช่น "X เรียกอะไรบ้าง") จะแสดงแค่ชั้น 1 พอ — เหตุผล: ข้อมูลชั้นลึกที่มีชื่อฟังก์ชันเยอะๆ (โดยเฉพาะ
        ฟังก์ชันภายในของกราฟเอง) ทำให้ context ยาวและซับซ้อนเกินจำเป็นสำหรับคำถามง่ายๆ ซึ่งจะไปลดความแม่นยำ
        ของโมเดลขนาดเล็ก (7B) และทำให้ตอบช้าลงโดยไม่จำเป็น
        """
        if depth is None:
            impact_keywords = ("กระทบ", "impact", "ถ้าแก้", "หากแก้", "ผลกระทบ", "affect")
            depth = 3 if any(k in query.lower() for k in impact_keywords) else 1
        matches = self.find_matches(query, limit=top_k)
        return [{"text": self.describe(n, depth), "source": n.file, "node": n.name} for n in matches]
