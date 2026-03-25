# 📝 Command Editing Feature

## Overview

The terminal AI now automatically detects placeholders in suggested commands and offers to let you edit them before execution. This ensures that training data contains real, executable commands instead of templates with placeholders.

---

## How It Works

### Automatic Placeholder Detection

The system detects common placeholder patterns:
- `<placeholder>` - Angle brackets (e.g., `<file>`, `<path>`)
- `[option]` - Square brackets
- `{value}` - Curly braces
- `...` - Ellipsis
- `YOUR_*` - Uppercase prefixes (e.g., `YOUR_API_KEY`)
- `REPLACE_*` - Replace prefixes

### Interactive Flow

#### Example 1: Command with Placeholder

```bash
$ term-ai "how to delete a file from git but keep it in gitignore"

🧠 NL modu: "how to delete a file from git but keep it in gitignore"
🔍 Arama yapılıyor...
   → git/untrack [vektör (0.92)] (38ms)

🤖 [local/qwen2.5-coder:7b] komut üretiliyor...

▶ Önerilen Komut: git rm --cached <file>
▶ Toplam süre: 1.1s | local/qwen2.5-coder:7b

⚠️  Komutta placeholder tespit edildi: git rm --cached <file>
▶ Düzenlemek ister misin? [e/h]: e

📝 Komutu düzenle (Enter = değiştirme, Ctrl+C = iptal):
   Orijinal: git rm --cached <file>
   Yeni:     git rm --cached IMPROVEMENT_PLAN.md

▶ Düzenlenmiş Komut: git rm --cached IMPROVEMENT_PLAN.md
▶ Çalıştırayım mı? [e/h]: e

[command executes]
✅ Komut training data'ya eklendi
```

**Result in training_data.json:**
```json
{
  "instruction": "how to delete a file from git but keep it in gitignore",
  "output": "git rm --cached IMPROVEMENT_PLAN.md",
  "tool": "git",
  "timestamp": "2026-03-25T14:23:00+03:00",
  "source": "user_approved"
}
```

#### Example 2: No Placeholders

```bash
$ term-ai "show git status"

🧠 NL modu: "show git status"
🔍 Arama yapılıyor...
   → git/status [vektör (0.95)] (42ms)

🤖 [local/qwen2.5-coder:7b] komut üretiliyor...

▶ Önerilen Komut: git status
▶ Toplam süre: 0.8s | local/qwen2.5-coder:7b
▶ Çalıştırayım mı? [e/h]: e

[command executes - no edit prompt because no placeholders]
✅ Komut training data'ya eklendi
```

---

## Benefits

### 1. **Better Training Data Quality**

**Before (with placeholders):**
```json
{
  "instruction": "find python files",
  "output": "find . -name <pattern>",  ❌ Not executable
  "tool": "find"
}
```

**After (with editing):**
```json
{
  "instruction": "find python files",
  "output": "find . -name '*.py'",  ✅ Real, executable command
  "tool": "find"
}
```

### 2. **Immediate Execution**

- Edit the command to make it executable
- Run it immediately
- Save the working version for training

### 3. **Learning Real Patterns**

The model learns from actual commands you run, not templates:
- Real file paths
- Actual flag combinations
- Your specific use cases

---

## User Options

### Option 1: Edit the Command

```
⚠️  Komutta placeholder tespit edildi: stanctl backend apply --core-base-domain <domain>
▶ Düzenlemek ister misin? [e/h]: e

📝 Komutu düzenle:
   Orijinal: stanctl backend apply --core-base-domain <domain>
   Yeni:     stanctl backend apply --core-base-domain prod.example.com
```

### Option 2: Skip Editing

```
⚠️  Komutta placeholder tespit edildi: git rm --cached <file>
▶ Düzenlemek ister misin? [e/h]: h

▶ Çalıştırayım mı? [e/h]: h
İptal edildi.
```

The command with placeholder is **not saved** to training data if you skip editing and don't execute it.

### Option 3: No Placeholders Detected

If the command has no placeholders, you go straight to execution:
```
▶ Önerilen Komut: git status
▶ Çalıştırayım mı? [e/h]: e
```

---

## Placeholder Detection Examples

### ✅ Detected Placeholders

```bash
git rm --cached <file>                    # <file>
find . -name <pattern>                    # <pattern>
docker run -p [port]:80 nginx             # [port]
kubectl get pods -n {namespace}           # {namespace}
curl -H "Authorization: YOUR_API_KEY"     # YOUR_API_KEY
sed 's/REPLACE_THIS/with_that/g'          # REPLACE_THIS
tar -xzf archive.tar.gz ...               # ...
```

### ❌ Not Detected (False Positives Avoided)

```bash
echo "Hello <World>"                      # In quotes, likely intentional
grep -E '<html>'                          # Regex pattern
awk '{print $1}'                          # AWK syntax
```

---

## Advanced Usage

### Editing Complex Commands

For multi-line or complex commands, you can:

1. **Edit inline** (simple changes):
   ```
   Yeni: stanctl backend apply --install-type production --core-base-domain prod.example.com
   ```

2. **Cancel and re-run** (major changes):
   - Press Ctrl+C to cancel
   - Re-run with more specific query
   - Get better suggestion

### IBM Tool Examples

#### Example: stanctl with placeholders

```bash
$ term-ai "install instana backend"

▶ Önerilen Komut: stanctl backend apply --install-type <type> --core-base-domain <domain>

⚠️  Komutta placeholder tespit edildi
▶ Düzenlemek ister misin? [e/h]: e

📝 Komutu düzenle:
   Yeni: stanctl backend apply --install-type production --core-base-domain instana.prod.company.com

▶ Düzenlenmiş Komut: stanctl backend apply --install-type production --core-base-domain instana.prod.company.com
▶ Çalıştırayım mı? [e/h]: e
```

**Saved to training data:**
```json
{
  "instruction": "install instana backend",
  "output": "stanctl backend apply --install-type production --core-base-domain instana.prod.company.com",
  "tool": "stanctl"
}
```

---

## Impact on Training Data

### Before This Feature

**Problem:** Training data contained unusable templates
```json
[
  {"instruction": "delete git file", "output": "git rm --cached <file>"},
  {"instruction": "find files", "output": "find . -name <pattern>"},
  {"instruction": "install backend", "output": "stanctl backend apply --install-type <type>"}
]
```

**Result:** Model learns to generate placeholders, not real commands

### After This Feature

**Solution:** Training data contains real, executable commands
```json
[
  {"instruction": "delete git file", "output": "git rm --cached README.md"},
  {"instruction": "find files", "output": "find . -name '*.py'"},
  {"instruction": "install backend", "output": "stanctl backend apply --install-type production"}
]
```

**Result:** Model learns actual patterns and real-world usage

---

## Best Practices

### 1. Always Edit Placeholders

When you see a placeholder prompt, take the time to fill it in correctly:
- ✅ Use real file names
- ✅ Use actual domain names
- ✅ Use correct flag values

### 2. Be Specific in Queries

**Instead of:**
```bash
term-ai "delete a file from git"
```

**Try:**
```bash
term-ai "delete IMPROVEMENT_PLAN.md from git but keep in gitignore"
```

This often results in fewer placeholders in the suggestion.

### 3. Review Before Executing

Even after editing, review the command:
- Check syntax
- Verify paths
- Confirm flags

### 4. Cancel If Unsure

If you're not sure how to fill in a placeholder:
- Press Ctrl+C to cancel
- Research the correct value
- Re-run with more context

---

## Technical Details

### Placeholder Detection Logic

```go
func hasPlaceholders(cmd string) bool {
    placeholderPatterns := []string{
        "<", ">",           // <file>, <path>
        "[", "]",           // [option]
        "{", "}",           // {value}
        "...",              // ellipsis
        "YOUR_",            // YOUR_API_KEY
        "REPLACE_",         // REPLACE_THIS
    }
    
    for _, pattern := range placeholderPatterns {
        if strings.Contains(cmd, pattern) {
            return true
        }
    }
    return false
}
```

### Edit Flow

```go
func askToEdit(cmd string) (string, bool) {
    if !hasPlaceholders(cmd) {
        return cmd, false  // No placeholders, skip editing
    }
    
    fmt.Printf("\n⚠️  Komutta placeholder tespit edildi: %s\n", cmd)
    fmt.Printf("▶ Düzenlemek ister misin? [e/h]: ")
    
    if readYesNo() {
        edited := editCommand(cmd)
        return edited, true
    }
    
    return cmd, false
}
```

---

## Troubleshooting

### Issue: False Positive Detection

**Problem:** Command has `<` or `>` but it's not a placeholder (e.g., redirect)

**Example:**
```bash
echo "test" > output.txt  # > is redirect, not placeholder
```

**Solution:** 
- Choose "no" when asked to edit
- The command will execute as-is
- Future versions may improve detection

### Issue: Can't Edit Multi-line Commands

**Problem:** Complex commands are hard to edit inline

**Solution:**
- Cancel (Ctrl+C)
- Break into simpler queries
- Or use `--chat` mode for complex scenarios

### Issue: Edited Command Still Has Issues

**Problem:** After editing, command still doesn't work

**Solution:**
- Don't approve it (choose "h" when asked to execute)
- It won't be saved to training data
- Refine your query and try again

---

## Summary

✅ **Automatic Detection**: Finds placeholders in suggested commands
✅ **Interactive Editing**: Lets you fill in real values
✅ **Better Training Data**: Saves executable commands, not templates
✅ **Immediate Execution**: Edit and run in one flow
✅ **Quality Control**: Only approved, working commands are saved

This feature ensures your training data is high-quality and ready for fine-tuning, with real commands from your actual workflow instead of generic templates.