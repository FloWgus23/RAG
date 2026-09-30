"""
fetch_dataset.py — ดึง dataset (SmolML: https://github.com/rodmarkun/SmolML) มาใส่โฟลเดอร์ my_dataset/

วิธีใช้:
    python fetch_dataset.py                    # ดาวน์โหลด ZIP จาก GitHub อัตโนมัติ
    python fetch_dataset.py --zip SmolML-main.zip   # ใช้ ZIP ที่โหลดไว้แล้ว (Code -> Download ZIP)
    python fetch_dataset.py --with-tests       # เอาโฟลเดอร์ tests/*.py มาด้วย (ค่าเริ่มต้นไม่เอา)

ทำไมต้อง "แบน" ชื่อไฟล์:
    หน้า Upload ของ Streamlit ส่งมาแค่ชื่อไฟล์ (ไม่มีโฟลเดอร์) และ graph_rag.py ตั้งชื่อโมดูลจากชื่อไฟล์
    ดังนั้นสคริปต์นี้จึงตั้งชื่อไฟล์ตาม path ของแพ็กเกจ เช่น
        smolml/core/ml_array.py      ->  smolml.core.ml_array.py     (โมดูล smolml.core.ml_array)
        smolml/core/__init__.py      ->  smolml.core.py              (โมดูล smolml.core)
        smolml/core/README.md        ->  smolml.core.README.md
    ทำให้ `from smolml.core import X` และ `from .ml_array import Y` ถูกโยงเข้าโมดูลที่ถูกต้องใน Graph mode
    และไม่มีชื่อไฟล์ซ้ำกัน (เช่น __init__.py หลายอัน)

หมายเหตุ: สคริปต์จะลบไฟล์เดิมในโฟลเดอร์ปลายทางก่อนเขียนใหม่
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "rodmarkun/SmolML"
BRANCH = "main"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
PACKAGE_DIR = "smolml"
TESTS_DIR = "tests"


def flat_name(rel_path: str) -> str:
    """แปลง path ในรีโป (ตัด root แล้ว) เป็นชื่อไฟล์แบนแบบ dotted"""
    parts = rel_path.split("/")
    if parts[-1] == "__init__.py":          # __init__.py ของแพ็กเกจ = ตัวแพ็กเกจเอง
        parts = parts[:-1]
        return ".".join(parts) + ".py"
    return ".".join(parts)


def wanted(rel_path: str, with_tests: bool) -> bool:
    parts = rel_path.split("/")
    name = parts[-1]
    if rel_path == "README.md":
        return True
    if parts[0] == PACKAGE_DIR:
        return name.endswith((".py", ".md"))
    if with_tests and parts[0] == TESTS_DIR:
        return name.endswith(".py")
    return False


def load_zip(zip_path: str | None) -> zipfile.ZipFile:
    if zip_path:
        return zipfile.ZipFile(zip_path)
    print(f"กำลังดาวน์โหลด {ZIP_URL} ...")
    with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def main() -> int:
    ap = argparse.ArgumentParser(description="ดึง SmolML มาเป็น dataset ของ Flowchat")
    ap.add_argument("--zip", help="path ของไฟล์ ZIP ที่โหลดไว้แล้ว (ไม่ระบุ = โหลดจาก GitHub)")
    ap.add_argument("--out", default="my_dataset", help="โฟลเดอร์ปลายทาง (ค่าเริ่มต้น my_dataset)")
    ap.add_argument("--with-tests", action="store_true", help="รวม tests/*.py ด้วย")
    args = ap.parse_args()

    zf = load_zip(args.zip)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    root = names[0].split("/")[0] + "/"          # เช่น "SmolML-main/"

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for old in out.iterdir():                    # ล้างของเดิม (เฉพาะไฟล์)
        if old.is_file():
            old.unlink()

    written, skipped_empty = [], 0
    for member in names:
        if not member.startswith(root):
            continue
        rel = member[len(root):]
        if not wanted(rel, args.with_tests):
            continue
        data = zf.read(member)
        if not data.strip():                     # __init__.py ที่ว่างเปล่าไม่มีประโยชน์
            skipped_empty += 1
            continue
        target = out / flat_name(rel)
        target.write_bytes(data)
        written.append(target)

    if not written:
        print("ไม่พบไฟล์ที่ต้องการใน ZIP — โครงสร้างรีโปอาจเปลี่ยน", file=sys.stderr)
        return 1

    total_kb = sum(p.stat().st_size for p in written) / 1024
    print(f"เขียน {len(written)} ไฟล์ ({total_kb:.0f} KB) ลง {out}/  (ข้าม __init__.py ว่าง {skipped_empty} ไฟล์)")
    for p in sorted(written):
        print("  ", p.name)
    print("\nขั้นต่อไป: streamlit run app.py แล้วอัปโหลดไฟล์ทั้งหมดใน my_dataset/ ที่หน้า Knowledge Base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
