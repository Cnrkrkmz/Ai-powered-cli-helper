import os
import re
import json
import urllib.request
import tempfile

# ─── Ayarlar ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "knowledge/open_dataset"

# Klasör yapısına alınacak araçlar
TARGET_TOOLS = [
    "find", "tar", "grep", "ls", "git", "docker",
    "kubectl", "awk", "sed", "chmod", "chown", "curl",
    "ssh", "rsync", "ps", "kill", "systemctl"
]

# Bu araçlar için 2. kelime sub-command sayılır (flag değil)
SUBCOMMAND_TOOLS = ["git", "docker", "kubectl", "systemctl"]

# Her alt kategori için max örnek sayısı
MAX_EXAMPLES = 15

# ─── Dataset ──────────────────────────────────────────────────────────────────

print("⏳ Dataset indiriliyor: nl2bash")

# Check if local files exist
nl_file = "nl2bash_pairs.txt"
cmd_file = "nl2bash_data.txt"

if not os.path.exists(nl_file) or not os.path.exists(cmd_file):
    print("   ⚠️  Dosyalar bulunamadı. İndiriliyor...")
    print("   → nl2bash_pairs.txt (natural language)")
    os.system('curl -s -L -o nl2bash_pairs.txt "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl"')
    print("   → nl2bash_data.txt (commands)")
    os.system('curl -s -L -o nl2bash_data.txt "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"')
    print()

# Load both files (they are parallel - same line numbers match)
try:
    with open(nl_file, 'r', encoding='utf-8') as f:
        nl_lines = f.read().strip().split('\n')
    
    with open(cmd_file, 'r', encoding='utf-8') as f:
        cmd_lines = f.read().strip().split('\n')
    
    # Pair them up
    all_items = []
    for nl_desc, cmd in zip(nl_lines, cmd_lines):
        nl_desc = nl_desc.strip()
        cmd = cmd.strip()
        if nl_desc and cmd:
            all_items.append({
                'nl': nl_desc,
                'bash': cmd
            })
    
    print(f"   → Toplam {len(all_items)} satır yüklendi\n")
    
except Exception as e:
    print(f"   ❌ Dosya okuma hatası: {e}")
    print("\n💡 Manuel indirme:")
    print("   curl -L -o nl2bash_pairs.txt 'https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl'")
    print("   curl -L -o nl2bash_data.txt 'https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm'")
    exit(1)

# ─── Veriyi Yapılandır ────────────────────────────────────────────────────────

print("🧠 Hiyerarşik yapıya göre analiz ediliyor...")

# Format: { "git": { "push": [(nl, cmd), ...], "general": [...] } }
structured = {}

for item in all_items:
    cmd     = str(item.get("bash", "")).strip()
    nl_desc = str(item.get("nl", "")).strip()

    if not cmd or not nl_desc:
        continue

    parts = cmd.split()
    base_tool = parts[0].lower()

    if base_tool not in TARGET_TOOLS:
        continue

    # Sub-category tespiti
    sub = "general"
    if len(parts) > 1:
        part2 = parts[1].lower()

        if base_tool in SUBCOMMAND_TOOLS:
            # git push → sub="push"
            if re.match(r"^[a-z][a-z0-9-]*$", part2) and not part2.startswith("-"):
                sub = part2
        else:
            # find -name → sub="name"  (2-8 harf, sadece flag kısmı)
            if part2.startswith("-"):
                clean = re.sub(r"^-+", "", part2)
                if re.match(r"^[a-z]{2,8}$", clean):
                    sub = clean

    if base_tool not in structured:
        structured[base_tool] = {}
    if sub not in structured[base_tool]:
        structured[base_tool][sub] = set()

    if len(structured[base_tool][sub]) < MAX_EXAMPLES:
        structured[base_tool][sub].add((nl_desc, cmd))

# ─── Dosyaları Yaz ───────────────────────────────────────────────────────────

print(f"📁 Dosyalar yazılıyor → {OUTPUT_DIR}/\n")

saved_tools = 0
saved_files = 0

for tool, sub_categories in sorted(structured.items()):
    # En az 1 kategoride 2+ örnek olan araçları al
    non_empty = {k: v for k, v in sub_categories.items() if len(v) >= 2}
    if not non_empty:
        continue

    tool_dir = os.path.join(OUTPUT_DIR, tool)
    os.makedirs(tool_dir, exist_ok=True)
    saved_tools += 1

    # FILES satırı için sub-category eşlemesi
    file_map_parts = [f"{sub}={sub}" for sub in sorted(non_empty.keys())]

    # _meta.txt — vektör sistemine uygun format
    meta_path = os.path.join(tool_dir, "_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"# DESCRIPTION: {tool} command usage examples from NL2Bash dataset\n")
        f.write(f"# KEYWORDS: {tool}, " + ", ".join(sorted(non_empty.keys())) + "\n")
        f.write(f"# FILES: " + " | ".join(file_map_parts) + "\n")

    # Alt kategori .txt dosyaları
    for sub, examples in sorted(non_empty.items()):
        file_path = os.path.join(tool_dir, f"{sub}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {tool} — {sub} examples\n\n")
            for nl_desc, cmd in sorted(examples):
                f.write(f"Task: {nl_desc}\n")
                f.write(f"Command: {cmd}\n\n")
        saved_files += 1
        print(f"   ✅ {tool}/{sub}.txt ({len(examples)} örnek)")

# ─── Özet ─────────────────────────────────────────────────────────────────────

print(f"""
✅ Tamamlandı!
   {saved_tools} araç klasörü
   {saved_files} .txt dosyası
   → {OUTPUT_DIR}/

Sonraki adım:
   term-ai --index
""")