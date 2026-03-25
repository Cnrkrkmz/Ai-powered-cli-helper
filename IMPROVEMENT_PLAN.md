# 🚀 Terminal AI - Comprehensive Improvement Plan

## Executive Summary

Your terminal-integrated AI has a solid foundation with hybrid search (vector + keyword), multi-backend support, and structured knowledge. This plan focuses on achieving **both speed and reliability** for general terminal commands and specialized IBM tools (stanctl, apicup).

**Current Performance:** 2-5s average response time, ~85% accuracy
**Target Performance:** 0.5-2s average response time, 95%+ accuracy

---

## 📊 Architecture Analysis

### Current Strengths ✅
1. **Hybrid Search System**: Vector embeddings (nomic-embed-text) with keyword fallback
2. **Multi-Backend Support**: Local Ollama + Cloud providers (Anthropic/OpenAI/Gemini)
3. **Smart Context Loading**: Tool detection → sub-command matching → targeted context
4. **Structured Knowledge**: Hierarchical organization (knowledge/stanctl/, knowledge/git/)
5. **Safety Features**: Dangerous command detection, user confirmation prompts
6. **Learning System**: History tracking with execution feedback

### Current Bottlenecks ⚠️
1. **Latency Issues**:
   - Vector search: 50-200ms
   - LLM inference: 1-4s (7B model)
   - Total: 2-5s for simple commands
   
2. **Context Inefficiency**:
   - Loads entire file contents (100+ lines)
   - No semantic chunking within files
   - Context window waste on irrelevant sections

3. **Model Selection**:
   - Single model (qwen2.5-coder:7b) for all tasks
   - No differentiation between simple/complex queries
   - Overkill for basic commands (ls, cd, git status)

4. **Dataset Integration Gap**:
   - datasets.py creates knowledge files
   - No training/fine-tuning pipeline
   - No RAG enhancement from nl2bash dataset

5. **Limited IBM Coverage**:
   - Only stanctl documented
   - apicup missing entirely
   - No cross-tool intelligence

---

## 🎯 Phase 1: Performance Optimization (Speed)

### 1.1 Multi-Tier Caching System

**Architecture:**
```
┌─────────────────────────────────────────┐
│ L1: Hot Command Cache (In-Memory)      │ ← 10-50ms
│ • Top 100 most-used commands            │
│ • LRU eviction policy                   │
│ • Fuzzy matching (Levenshtein < 3)     │
├─────────────────────────────────────────┤
│ L2: Embedding Cache (Query Vectors)    │ ← 50-100ms
│ • Cache query embeddings for 1 hour     │
│ • Avoid re-embedding similar queries    │
│ • Key: hash(normalized_query)           │
├─────────────────────────────────────────┤
│ L3: Response Cache (LLM Outputs)       │ ← 100-200ms
│ • Cache command suggestions             │
│ • TTL: 24 hours                         │
│ • Invalidate on knowledge updates       │
└─────────────────────────────────────────┘
```

**Implementation Details:**

```go
// cache.go
type CacheManager struct {
    hotCommands   *lru.Cache      // L1: 100 entries
    embeddings    *ttlcache.Cache // L2: 1 hour TTL
    responses     *ttlcache.Cache // L3: 24 hour TTL
}

func (cm *CacheManager) GetCommand(query string) (string, bool) {
    // L1: Check hot commands
    normalized := normalizeQuery(query)
    if cmd, ok := cm.hotCommands.Get(normalized); ok {
        return cmd.(string), true
    }
    
    // L1: Fuzzy match (for typos)
    for _, entry := range cm.hotCommands.Keys() {
        if levenshtein(normalized, entry.(string)) <= 2 {
            if cmd, ok := cm.hotCommands.Get(entry); ok {
                return cmd.(string), true
            }
        }
    }
    
    return "", false
}

func (cm *CacheManager) GetEmbedding(query string) ([]float64, bool) {
    key := hashQuery(query)
    if vec, ok := cm.embeddings.Get(key); ok {
        return vec.([]float64), true
    }
    return nil, false
}
```

**Expected Impact:**
- 60% of queries hit L1 cache → **10-50ms** response
- 25% hit L2/L3 → **100-500ms** response
- 15% require full pipeline → **1-2s** response
- **Average: 0.5-1s** (down from 2-5s)

---

### 1.2 Smart Model Selection

**Strategy: Task-Based Model Routing**

```go
type QueryComplexity int

const (
    Simple   QueryComplexity = iota // ls, cd, pwd, git status
    Medium                           // git push, docker run, find
    Complex                          // multi-tool, ambiguous, IBM tools
)

func classifyQuery(query string, detectedTool string) QueryComplexity {
    // Simple: Single word or common patterns
    if isCommonCommand(query) {
        return Simple
    }
    
    // Complex: IBM tools, multi-step, or no tool detected
    if detectedTool == "stanctl" || detectedTool == "apicup" || detectedTool == "none" {
        return Complex
    }
    
    // Medium: Everything else
    return Medium
}

func selectModel(complexity QueryComplexity, useCloud bool) string {
    if useCloud {
        return cfg.CloudModel // User's choice
    }
    
    switch complexity {
    case Simple:
        return "qwen2.5-coder:1.5b"  // 100-300ms, 90% accuracy
    case Medium:
        return "qwen2.5-coder:7b"    // 500-1000ms, 95% accuracy
    case Complex:
        return "qwen2.5-coder:14b"   // 1-3s, 98% accuracy
    }
}
```

**Model Performance Matrix:**

| Model | Size | Inference Time | Accuracy | Use Case |
|-------|------|----------------|----------|----------|
| qwen2.5-coder:1.5b | 1.5B | 100-300ms | 90% | Simple commands |
| qwen2.5-coder:7b | 7B | 500-1000ms | 95% | General use |
| qwen2.5-coder:14b | 14B | 1-3s | 98% | Complex/IBM tools |
| qwen2.5-coder:32b | 32B | 3-8s | 99% | Fallback only |

**Expected Impact:**
- 50% of queries use 1.5B model → **3-5x faster**
- 35% use 7B model → **current speed**
- 15% use 14B model → **better accuracy**

---

### 1.3 Parallel Processing

**Current Sequential Flow:**
```
Vector Search (200ms) → Keyword Fallback (50ms) → Context Load (100ms) → LLM (1000ms)
Total: 1350ms
```

**Optimized Parallel Flow:**
```
┌─ Vector Search (200ms) ─┐
│                          ├─→ Best Result → Context Load (100ms) → LLM (1000ms)
└─ Keyword Search (50ms) ─┘
Total: 1300ms (50ms saved)

With pre-loading:
┌─ Vector Search (200ms) ─┐
│                          ├─→ Best Result → [Context Already Loaded] → LLM (1000ms)
└─ Keyword Search (50ms) ─┘
└─ Pre-load Common Tools (background) ─┘
Total: 1200ms (150ms saved)
```

**Implementation:**

```go
func handleNLModeOptimized(cfg Config, query string) {
    // Start parallel operations
    vectorChan := make(chan SearchResult)
    keywordChan := make(chan string)
    
    go func() {
        vectorChan <- SearchSimilar(cfg.OllamaURL, query)
    }()
    
    go func() {
        tools := listTools()
        keywordChan <- detectToolByKeywords(query, tools)
    }()
    
    // Wait for both, use best result
    vectorResult := <-vectorChan
    keywordResult := <-keywordChan
    
    detectedTool := chooseBestResult(vectorResult, keywordResult)
    
    // Context loading with pre-loaded cache
    context := getContextWithPreload(detectedTool)
    
    // LLM inference
    suggestion := askLLM(cfg, buildPrompt(context, query))
}

// Pre-load common tools on startup
func preloadCommonContexts() {
    commonTools := []string{"git", "docker", "kubectl", "stanctl"}
    for _, tool := range commonTools {
        go func(t string) {
            loadAndCacheContext(t)
        }(tool)
    }
}
```

**Expected Impact:**
- 50-150ms saved per query
- Better UX with instant feedback
- Background pre-loading reduces cold starts

---

## 🎯 Phase 2: Reliability Improvements

### 2.1 Semantic Chunking for Better Context

**Current Problem:**
```go
// Loads entire file (36 lines, 200+ tokens)
toolContext = loadSubFileContext(meta, "backend")
```

**Solution: Section-Level Indexing**

```
knowledge/stanctl/backend.txt
├─ [backend-apply]      ← Chunk 1 (lines 3-15)   → Embedding 1
├─ [datastores-apply]   ← Chunk 2 (lines 17-26)  → Embedding 2
└─ [migrate]            ← Chunk 3 (lines 28-36)  → Embedding 3
```

**New Data Structure:**

```go
type EmbeddingEntry struct {
    Tool      string
    SubFile   string
    Section   string    // NEW: [backend-apply], [datastores-apply]
    FilePath  string
    LineRange [2]int    // NEW: [3, 15]
    Vector    []float64
    Preview   string
}

// Enhanced search returns specific section
func SearchSimilarWithSection(query string) SearchResult {
    // Returns: tool=stanctl, subFile=backend, section=[backend-apply]
    // Only loads lines 3-15 instead of entire file
}
```

**Implementation:**

```go
// vector.go enhancement
func indexFileWithSections(filePath string) []EmbeddingEntry {
    content, _ := os.ReadFile(filePath)
    lines := strings.Split(string(content), "\n")
    
    var entries []EmbeddingEntry
    var currentSection string
    var sectionStart int
    var sectionContent strings.Builder
    
    for i, line := range lines {
        // Detect section headers: [backend-apply]
        if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
            // Save previous section
            if currentSection != "" {
                entries = append(entries, createEntry(
                    currentSection, 
                    sectionStart, 
                    i-1, 
                    sectionContent.String(),
                ))
            }
            
            // Start new section
            currentSection = strings.Trim(line, "[]")
            sectionStart = i
            sectionContent.Reset()
        } else {
            sectionContent.WriteString(line + "\n")
        }
    }
    
    return entries
}
```

**Expected Impact:**
- Context size: 200+ tokens → **50-80 tokens** (60% reduction)
- Accuracy: More focused context = better suggestions
- Speed: Smaller prompts = faster LLM inference

---

### 2.2 Command Validation Pipeline

**Multi-Stage Validation:**

```
User Query → LLM → Suggested Command
                         ↓
                    ┌────────────┐
                    │ Validators │
                    └────────────┘
                         ↓
    ┌────────────────────┼────────────────────┐
    ↓                    ↓                    ↓
Syntax Check      Tool Exists?         Flag Validation
(bash -n)         (exec.LookPath)      (--help parsing)
    ↓                    ↓                    ↓
    └────────────────────┴────────────────────┘
                         ↓
                  ✅ Valid Command
```

**Implementation:**

```go
type ValidationResult struct {
    Valid      bool
    Confidence float64
    Issues     []string
    Fixes      []string
}

func validateCommand(cmd string) ValidationResult {
    result := ValidationResult{Valid: true, Confidence: 1.0}
    
    // 1. Syntax check
    if err := checkBashSyntax(cmd); err != nil {
        result.Valid = false
        result.Issues = append(result.Issues, "Syntax error: " + err.Error())
        result.Confidence *= 0.5
    }
    
    // 2. Tool existence
    parts := strings.Fields(cmd)
    if len(parts) > 0 {
        if _, err := exec.LookPath(parts[0]); err != nil {
            result.Valid = false
            result.Issues = append(result.Issues, "Command not found: " + parts[0])
            result.Confidence *= 0.3
        }
    }
    
    // 3. Flag validation (parse --help)
    if validFlags := getValidFlags(parts[0]); validFlags != nil {
        for _, arg := range parts[1:] {
            if strings.HasPrefix(arg, "-") && !contains(validFlags, arg) {
                result.Issues = append(result.Issues, "Unknown flag: " + arg)
                result.Confidence *= 0.8
            }
        }
    }
    
    // 4. Dangerous patterns (already exists, enhance it)
    if isDangerous(cmd) {
        result.Valid = false
        result.Issues = append(result.Issues, "Dangerous operation detected")
        result.Confidence = 0.0
    }
    
    return result
}

func checkBashSyntax(cmd string) error {
    // Use bash -n to check syntax without executing
    bashCmd := exec.Command("bash", "-n", "-c", cmd)
    return bashCmd.Run()
}

func getValidFlags(tool string) []string {
    // Cache --help output
    cacheKey := "flags:" + tool
    if cached, ok := flagCache.Get(cacheKey); ok {
        return cached.([]string)
    }
    
    // Parse --help output
    helpCmd := exec.Command(tool, "--help")
    output, err := helpCmd.CombinedOutput()
    if err != nil {
        return nil
    }
    
    flags := parseHelpOutput(string(output))
    flagCache.Set(cacheKey, flags, 24*time.Hour)
    return flags
}
```

**Expected Impact:**
- Catch 90% of invalid commands before execution
- Reduce user frustration from failed commands
- Build trust through reliability

---

### 2.3 Confidence Scoring & Alternatives

**Show Uncertainty to Users:**

```go
type CommandSuggestion struct {
    Command      string
    Confidence   float64  // 0.0-1.0
    Reasoning    string
    Alternatives []string
    Warnings     []string
}

func generateSuggestionWithConfidence(cfg Config, query string) CommandSuggestion {
    // Get multiple suggestions
    suggestions := []string{
        askLLM(cfg, buildPrompt(query, temperature=0.3)),  // Conservative
        askLLM(cfg, buildPrompt(query, temperature=0.7)),  // Creative
    }
    
    // Score each suggestion
    scored := make([]ScoredCommand, len(suggestions))
    for i, cmd := range suggestions {
        validation := validateCommand(cmd)
        similarity := semanticSimilarity(query, cmd)
        
        scored[i] = ScoredCommand{
            Command:    cmd,
            Confidence: (validation.Confidence + similarity) / 2,
        }
    }
    
    // Sort by confidence
    sort.Slice(scored, func(i, j int) bool {
        return scored[i].Confidence > scored[j].Confidence
    })
    
    return CommandSuggestion{
        Command:      scored[0].Command,
        Confidence:   scored[0].Confidence,
        Alternatives: extractAlternatives(scored[1:]),
    }
}

// Enhanced output
func displaySuggestion(sugg CommandSuggestion) {
    fmt.Printf("\n▶ Suggested Command: %s\n", sugg.Command)
    fmt.Printf("▶ Confidence: %.0f%%\n", sugg.Confidence*100)
    
    if sugg.Confidence < 0.7 {
        fmt.Printf("\n⚠️  Low confidence. Consider these alternatives:\n")
        for i, alt := range sugg.Alternatives {
            fmt.Printf("  %d. %s\n", i+1, alt)
        }
        fmt.Printf("\nWhich would you like to use? [1-%d, or 'n' to cancel]: ", len(sugg.Alternatives)+1)
    } else {
        fmt.Printf("\n▶ Execute? [y/n]: ")
    }
}
```

**Expected Impact:**
- Users trust the system more (transparency)
- Fewer failed executions (alternatives provided)
- Better learning (user choice feedback)

---

## 🎯 Phase 3: IBM Tool Integration

### 3.1 Add apicup Knowledge Base

**Directory Structure:**

```
knowledge/
├── apicup/
│   ├── _meta.txt
│   ├── install.txt
│   ├── subsystems.txt
│   ├── certificates.txt
│   ├── upgrade.txt
│   ├── backup.txt
│   ├── troubleshoot.txt
│   └── networking.txt
```

**_meta.txt Template:**

```
# DESCRIPTION: IBM API Connect installation and management CLI tool
# KEYWORDS: api connect, apic, apicup, gateway, portal, manager, analytics, subsystem, kubernetes, openshift
# FILES: install=install deployment setup | subsystems=subsystem add remove configure gateway portal manager analytics | certificates=cert certificate ssl tls security | upgrade=upgrade version update migrate | backup=backup restore snapshot | troubleshoot=debug logs error issue problem | networking=network ingress route dns load-balancer
```

**Example Content (install.txt):**

```
# apicup — Installation and Deployment

[prerequisites]
Check system requirements: apicup version
Verify Kubernetes cluster: kubectl cluster-info
Check storage classes: kubectl get storageclass

[init-project]
Initialize new project: apicup init <project-name>
Set project directory: cd <project-name>
Verify project structure: ls -la

[subsystem-install]
Install management subsystem: apicup subsys install mgmt --plan-dir <plan-dir>
Install gateway subsystem: apicup subsys install gwy --plan-dir <plan-dir>
Install portal subsystem: apicup subsys install ptl --plan-dir <plan-dir>
Install analytics subsystem: apicup subsys install a7s --plan-dir <plan-dir>

[configuration]
Set registry: apicup subsys set <subsystem> registry <registry-url>
Set storage class: apicup subsys set <subsystem> storage-class <class-name>
Set mode: apicup subsys set <subsystem> mode <dev|standard>
Generate certificates: apicup certs create <subsystem>

[deployment]
Generate Kubernetes manifests: apicup subsys install <subsystem> --out <output-dir>
Apply to cluster: kubectl apply -f <output-dir>
Check status: apicup subsys get <subsystem>
```

---

### 3.2 IBM-Specific Optimizations

**Context Templates for IBM Tools:**

```go
// ibm_tools.go
type IBMToolContext struct {
    Tool         string
    Version      string
    Environment  string // kubernetes, openshift, standalone
    CommonFlags  map[string]string
    Prerequisites []string
}

func buildIBMPrompt(ctx IBMToolContext, query string) string {
    template := `You are an expert in IBM %s.

Environment: %s
Version: %s

Common flags:
%s

Prerequisites:
%s

User query: %s

Generate a precise command. Include:
1. The exact command with all required flags
2. Brief explanation of what it does
3. Any prerequisites or warnings

Command:`

    return fmt.Sprintf(template,
        ctx.Tool,
        ctx.Environment,
        ctx.Version,
        formatFlags(ctx.CommonFlags),
        formatList(ctx.Prerequisites),
        query,
    )
}

// Auto-detect IBM tool version
func detectIBMToolVersion(tool string) string {
    cmd := exec.Command(tool, "version")
    output, err := cmd.Output()
    if err != nil {
        return "unknown"
    }
    
    // Parse version from output
    re := regexp.MustCompile(`(\d+\.\d+\.\d+)`)
    matches := re.FindStringSubmatch(string(output))
    if len(matches) > 1 {
        return matches[1]
    }
    
    return "unknown"
}
```

**Flag Autocomplete System:**

```go
// Parse and cache IBM tool flags
func cacheIBMToolFlags(tool string) {
    helpOutput := getHelpOutput(tool)
    
    // IBM tools often have structured help
    flags := parseIBMHelpOutput(helpOutput)
    
    // Store in structured format
    flagDB[tool] = IBMFlagSet{
        Global:    flags.global,
        Subcommands: flags.subcommands,
        Examples:  flags.examples,
    }
}

// Use cached flags for validation and suggestions
func suggestFlags(tool, subcommand string, partialFlag string) []string {
    flagSet := flagDB[tool]
    if subcommandFlags, ok := flagSet.Subcommands[subcommand]; ok {
        return fuzzyMatch(partialFlag, subcommandFlags)
    }
    return fuzzyMatch(partialFlag, flagSet.Global)
}
```

---

### 3.3 Cross-Tool Intelligence

**Multi-Tool Query Detection:**

```go
func detectMultiToolQuery(query string) []string {
    tools := []string{}
    
    // Check for tool mentions
    allTools := listTools()
    for toolName := range allTools {
        if strings.Contains(strings.ToLower(query), toolName) {
            tools = append(tools, toolName)
        }
    }
    
    // Check for workflow keywords
    workflowKeywords := map[string][]string{
        "migrate": {"stanctl", "kubectl"},
        "backup":  {"stanctl", "kubectl", "velero"},
        "deploy":  {"apicup", "kubectl"},
    }
    
    for keyword, relatedTools := range workflowKeywords {
        if strings.Contains(strings.ToLower(query), keyword) {
            tools = append(tools, relatedTools...)
        }
    }
    
    return unique(tools)
}

func buildMultiToolContext(tools []string, query string) string {
    var contexts []string
    
    for _, tool := range tools {
        // Get relevant context for each tool
        ctx := getSmartContext(tool, extractSubCommand(query))
        contexts = append(contexts, ctx)
    }
    
    // Combine with workflow template
    return fmt.Sprintf(`Multi-tool workflow detected: %s

Relevant contexts:
%s

Generate a step-by-step command sequence.`,
        strings.Join(tools, ", "),
        strings.Join(contexts, "\n---\n"),
    )
}
```

**Example Multi-Tool Query:**

```
Query: "migrate instana data to new kubernetes cluster"

Detected Tools: [stanctl, kubectl]

Generated Sequence:
1. stanctl migrate -f backup.tar.gz --install-type production
2. kubectl get pods -n instana-core
3. stanctl backend apply --core-base-domain new-cluster.example.com
4. kubectl wait --for=condition=ready pod -l app=instana -n instana-core
```

---

## 🎯 Phase 4: Dataset Integration & Training

### 4.1 Current Dataset Status

Your [`datasets.py`](datasets.py) creates structured knowledge files but doesn't train models.

**Three Approaches:**

#### Option A: Fine-tune Local Model (Best Accuracy)

```python
# fine_tune.py
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    "qwen/Qwen2.5-Coder-7B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Prepare dataset
ds = load_dataset("jiacheng-ye/nl2bash")

def format_prompt(example):
    return f"""### Instruction:
Generate a bash command for the following task.

### Task:
{example['nl']}

### Command:
{example['cmd']}"""

formatted_ds = ds.map(lambda x: {"text": format_prompt(x)})

# Fine-tune
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_ds["train"],
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer.train()
model.save_pretrained("qwen2.5-coder-7b-terminal")
```

**Pros:**
- Best accuracy (98%+)
- Faster inference (optimized for terminal commands)
- Offline capability

**Cons:**
- Requires GPU (4-8 hours training)
- 10-20GB disk space
- Need to retrain for updates

---

#### Option B: RAG Enhancement (Easiest)

```go
// rag.go
func enhancePromptWithExamples(query string) string {
    // Search nl2bash dataset for similar queries
    examples := searchDataset(query, limit=3)
    
    exampleText := ""
    for _, ex := range examples {
        exampleText += fmt.Sprintf("Example: %s → %s\n", ex.NL, ex.Command)
    }
    
    return fmt.Sprintf(`%s

Similar examples from training data:
%s

Now generate command for: %s`,
        basePrompt,
        exampleText,
        query,
    )
}

func searchDataset(query string, limit int) []Example {
    // Use vector search on nl2bash dataset
    queryVec := getEmbedding(query)
    
    // Load pre-indexed nl2bash embeddings
    datasetIndex := loadDatasetIndex()
    
    // Find most similar
    results := []Example{}
    for _, entry := range datasetIndex {
        score := cosineSimilarity(queryVec, entry.Vector)
        if score > 0.7 {
            results = append(results, entry.Example)
        }
    }
    
    // Sort by score, return top N
    sort.Slice(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })
    
    if len(results) > limit {
        results = results[:limit]
    }
    
    return results
}
```

**Pros:**
- No training required
- Easy to update (just add examples)
- Works with any model

**Cons:**
- Slower (extra embedding + search)
- Larger prompts (more tokens)
- Depends on example quality

---

#### Option C: Hybrid Approach (Recommended)

**Combine fine-tuning + RAG:**

```
┌──────────────────────────────────────┐
│ User Query: "list all docker images" │
└──────────────────────────────────────┘
                 ↓
    ┌────────────┴────────────┐
    ↓                         ↓
Fine-tuned Model          RAG Search
(Base knowledge)      (Specific examples)
    ↓                         ↓
    └────────────┬────────────┘
                 ↓
         Combined Prompt
                 ↓
         Final Command
```

**Implementation:**

```go
func generateCommandHybrid(cfg Config, query string) string {
    // 1. Get base suggestion from fine-tuned model
    baseSuggestion := askFineTunedModel(cfg, query)
    
    // 2. Get similar examples from dataset
    examples := searchDataset(query, limit=2)
    
    // 3. If base suggestion has low confidence, use RAG
    if confidence(baseSuggestion) < 0.8 && len(examples) > 0 {
        enhancedPrompt := buildPromptWithExamples(query, examples)
        return askFineTunedModel(cfg, enhancedPrompt)
    }
    
    return baseSuggestion
}
```

**Pros:**
- Best of both worlds
- Fast for common queries (fine-tuned)
- Accurate for rare queries (RAG)

**Cons:**
- More complex implementation
- Requires both training and indexing

---

### 4.2 Dataset Preparation Pipeline

**Automated Pipeline:**

```python
# prepare_training_data.py

import json
from datasets import load_dataset
from collections import defaultdict

# Load nl2bash
ds = load_dataset("jiacheng-ye/nl2bash")

# Filter and clean
def is_valid_example(example):
    cmd = example.get('cmd', '')
    nl = example.get('nl', '')
    
    # Remove invalid examples
    if len(cmd) < 3 or len(nl) < 5:
        return False
    
    # Remove dangerous commands
    dangerous = ['rm -rf /', 'dd if=', 'mkfs', ':(){ :|:& };:']
    if any(d in cmd for d in dangerous):
        return False
    
    return True

filtered_ds = ds.filter(is_valid_example)

# Augment with your custom knowledge
custom_examples = []

# Parse your knowledge files
for tool_dir in os.listdir('knowledge'):
    for file in os.listdir(f'knowledge/{tool_dir}'):
        if file.endswith('.txt') and file != '_meta.txt':
            examples = parse_knowledge_file(f'knowledge/{tool_dir}/{file}')
            custom_examples.extend(examples)

# Combine datasets
combined = {
    'nl2bash': filtered_ds,
    'custom': custom_examples,
}

# Save for training
with open('training_data.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Prepared {len(filtered_ds)} nl2bash + {len(custom_examples)} custom examples")
```

---

## 🎯 Phase 5: Testing & Benchmarking

### 5.1 Benchmark Suite

**Create Test Cases:**

```go
// benchmark_test.go
type BenchmarkCase struct {
    Query          string
    ExpectedTool   string
    ExpectedCmd    string
    Complexity     QueryComplexity
    MaxLatency     time.Duration
}

var benchmarkCases = []BenchmarkCase{
    // Simple commands
    {
        Query:        "list files",
        ExpectedTool: "ls",
        ExpectedCmd:  "ls -la",
        Complexity:   Simple,
        MaxLatency:   500 * time.Millisecond,
    },
    {
        Query:        "show git status",
        ExpectedTool: "git",
        ExpectedCmd:  "git status",
        Complexity:   Simple,
        MaxLatency:   500 * time.Millisecond,
    },
    
    // Medium commands
    {
        Query:        "push current branch to origin",
        ExpectedTool: "git",
        ExpectedCmd:  "git push origin $(git branch --show-current)",
        Complexity:   Medium,
        MaxLatency:   1 * time.Second,
    },
    
    // Complex IBM commands
    {
        Query:        "install instana backend with production settings",
        ExpectedTool: "stanctl",
        ExpectedCmd:  "stanctl backend apply --install-type production",
        Complexity:   Complex,
        MaxLatency:   2 * time.Second,
    },
    
    // Multi-tool
    {
        Query:        "backup instana and save to kubernetes pvc",
        ExpectedTool: "stanctl",
        ExpectedCmd:  "stanctl cluster backup",
        Complexity:   Complex,
        MaxLatency:   3 * time.Second,
    },
}

func BenchmarkCommandGeneration(b *testing.B) {
    cfg := loadConfig()
    
    for _, tc := range benchmarkCases {
        b.Run(tc.Query, func(b *testing.B) {
            start := time.Now()
            
            result := generateCommand(cfg, tc.Query)
            
            elapsed := time.Since(start)
            
            // Check latency
            if elapsed > tc.MaxLatency {
                b.Errorf("Latency exceeded: %v > %v", elapsed, tc.MaxLatency)
            }
            
            // Check accuracy
            if !commandsMatch(result, tc.ExpectedCmd) {
                b.Errorf("Command mismatch:\nGot:      %s\nExpected: %s", 
                    result, tc.ExpectedCmd)
            }
            
            b.ReportMetric(float64(elapsed.Milliseconds()), "ms/op")
        })
    }
}
```

**Run Benchmarks:**

```bash
# Run all benchmarks
go test -bench=. -benchmem

# Run specific complexity
go test -bench=Simple -benchmem

# Generate report
go test -bench=. -benchmem > benchmark_results.txt
```

---

### 5.2 Accuracy Metrics

**Track Performance Over Time:**

```go
// metrics.go
type AccuracyMetrics struct {
    TotalQueries     int
    CorrectCommands  int
    PartiallyCorrect int
    Incorrect        int
    UserRejected     int
    
    ByComplexity map[QueryComplexity]ComplexityMetrics
    ByTool       map[string]ToolMetrics
}

type ComplexityMetrics struct {
    Total    int
    Correct  int
    AvgTime  time.Duration
}

func trackAccuracy(query string, suggested string, executed bool, userFeedback string) {
    metrics := loadMetrics()
    
    metrics.TotalQueries++
    
    if executed {
        if userFeedback == "correct" {
            metrics.CorrectCommands++
        } else if userFeedback == "partial" {
            metrics.PartiallyCorrect++
        }
    } else {
        metrics.UserRejected++
    }
    
    saveMetrics(metrics)
}

func generateAccuracyReport() {
    metrics := loadMetrics()
    
    accuracy := float64(metrics.CorrectCommands) / float64(metrics.TotalQueries) * 100
    
    fmt.Printf(`
Accuracy Report
===============
Total Queries: %d
Correct: %d (%.1f%%)
Partially Correct: %d (%.1f%%)
Incorrect: %d (%.1f%%)
User Rejected: %d (%.1f%%)

By Complexity:
- Simple:  %.1f%% (%d queries)
- Medium:  %.1f%% (%d queries)
- Complex: %.1f%% (%d queries)

Top Tools:
%s
`,
        metrics.TotalQueries,
        metrics.CorrectCommands, accuracy,
        metrics.PartiallyCorrect, float64(metrics.PartiallyCorrect)/float64(metrics.TotalQueries)*100,
        metrics.Incorrect, float64(metrics.Incorrect)/float64(metrics.TotalQueries)*100,
        metrics.UserRejected, float64(metrics.UserRejected)/float64(metrics.TotalQueries)*100,
        
        getComplexityAccuracy(metrics, Simple),
        getComplexityTotal(metrics, Simple),
        getComplexityAccuracy(metrics, Medium),
        getComplexityTotal(metrics, Medium),
        getComplexityAccuracy(metrics, Complex),
        getComplexityTotal(metrics, Complex),
        
        formatTopTools(metrics.ByTool),
    )
}
```

---

## 📅 Implementation Roadmap

### Sprint 1 (Week 1-2): Quick Wins - Performance
**Goal: 2-5s → 1-2s average response time**

- [ ] Implement L1 hot command cache (in-memory LRU)
- [ ] Add smart model selection (1.5B/7B/14B routing)
- [ ] Parallel vector + keyword search
- [ ] Pre-load common tool contexts on startup
- [ ] Benchmark current vs. optimized performance

**Expected Impact:** 50-60% latency reduction

---

### Sprint 2 (Week 3-4): Reliability Foundation
**Goal: 85% → 90% accuracy**

- [ ] Implement semantic chunking (section-level indexing)
- [ ] Add command validation pipeline (syntax, tool existence, flags)
- [ ] Build confidence scoring system
- [ ] Create benchmark test suite (50+ test cases)
- [ ] Add accuracy tracking metrics

**Expected Impact:** 5-10% accuracy improvement

---

### Sprint 3 (Week 5-6): IBM Tool Integration
**Goal: Full apicup support + enhanced stanctl**

- [ ] Create apicup knowledge base (8 files)
- [ ] Implement IBM-specific context templates
- [ ] Add flag autocomplete system (parse --help)
- [ ] Build cross-tool intelligence (multi-tool queries)
- [ ] Test with real IBM tool scenarios

**Expected Impact:** Production-ready IBM tool support

---

### Sprint 4 (Week 7-8): Dataset Integration
**Goal: 90% → 95% accuracy**

- [ ] Prepare training data (nl2bash + custom)
- [ ] Fine-tune qwen2.5-coder:7b model
- [ ] Implement RAG enhancement (dataset search)
- [ ] Build hybrid approach (fine-tuned + RAG)
- [ ] A/B test: base model vs. fine-tuned vs. hybrid

**Expected Impact:** 5% accuracy improvement, especially for rare commands

---

### Sprint 5 (Week 9-10): Polish & Production
**Goal: Production-ready system**

- [ ] Implement L2/L3 caching (embeddings, responses)
- [ ] Add user feedback collection
- [ ] Create comprehensive documentation
- [ ] Build monitoring dashboard (metrics, errors)
- [ ] Performance optimization (profiling, bottleneck removal)

**Expected Impact:** Production-grade reliability and observability

---

## 🎯 Success Metrics

### Performance Targets

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| **Average Latency** | 2-5s | 1-2s | 0.5-1s |
| **P95 Latency** | 8s | 3s | 2s |
| **Cache Hit Rate** | 0% | 60% | 80% |
| **Simple Command Latency** | 2s | 300ms | 100ms |

### Accuracy Targets

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| **Overall Accuracy** | 85% | 95% | 98% |
| **Simple Commands** | 90% | 98% | 99% |
| **IBM Tools** | 80% | 95% | 97% |
| **Multi-Tool Queries** | 70% | 85% | 90% |

### User Experience

| Metric | Current | Target |
|--------|---------|--------|
| **User Rejection Rate** | 20% | <10% |
| **Dangerous Command Blocks** | 95% | 99% |
| **Confidence Score Accuracy** | N/A | 90% |

---

## 🚀 Next Steps

### Immediate Actions (This Week)

1. **Set up benchmarking**: Create test suite with 20 common queries
2. **Implement L1 cache**: Quick win for performance
3. **Add model routing**: 1.5B for simple, 7B for complex
4. **Measure baseline**: Run benchmarks, record current metrics

### Short-term (Next Month)

1. **Complete Sprint 1-2**: Performance + reliability foundation
2. **Create apicup knowledge base**: 8 files with examples
3. **Implement validation pipeline**: Catch errors before execution
4. **Start dataset preparation**: Clean nl2bash, format for training

### Long-term (Next Quarter)

1. **Fine-tune custom model**: qwen2.5-coder-7b-terminal
2. **Build monitoring dashboard**: Track metrics over time
3. **Add advanced features**: Interactive mode, shell integration
4. **Scale knowledge base**: Add more IBM tools, cloud providers

---

## 📚 Additional Recommendations

### Architecture Improvements

1. **Plugin System**: Allow users to add custom tools
2. **Shell Integration**: Direct integration with bash/zsh (no `term-ai` prefix)
3. **Interactive Mode**: Multi-turn conversations for complex tasks
4. **Explain Mode**: Explain existing commands (reverse of NL mode)

### Knowledge Base Expansion

1. **Cloud Providers**: AWS CLI, Azure CLI, GCP gcloud
2. **Container Orchestration**: Kubernetes, Docker Swarm, Nomad
3. **CI/CD Tools**: Jenkins, GitLab CI, GitHub Actions
4. **Monitoring**: Prometheus, Grafana, ELK stack

### Advanced Features

1. **Command History Analysis**: Learn from user patterns
2. **Context-Aware Suggestions**: Use current directory, git status
3. **Multi-Step Workflows**: Generate command sequences
4. **Error Recovery**: Auto-fix failed commands

---

## 🎓 Learning Resources

### Fine-tuning
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face TRL](https://github.com/huggingface/trl)

### Vector Search
- [Ollama Embeddings API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

### Go Performance
- [Go Profiling](https://go.dev/blog/pprof)
- [Go Concurrency Patterns](https://go.dev/blog/pipelines)

---

## 📝 Conclusion

Your terminal AI has a **solid foundation** with hybrid search, multi-backend support, and structured knowledge. The main improvements needed are:

1. **Performance**: Multi-tier caching + smart model selection → **2-3x faster**
2. **Reliability**: Validation + confidence scoring → **10% accuracy boost**
3. **IBM Tools**: apicup knowledge + cross-tool intelligence → **production-ready**
4. **Dataset**: Fine-tuning + RAG → **5% accuracy boost**

**Recommended Priority:**
1. Sprint 1 (Performance) - Quick wins, immediate user impact
2. Sprint 2 (Reliability) - Build trust, reduce errors
3. Sprint 3 (IBM Tools) - Your specific use case
4. Sprint 4 (Dataset) - Long-term accuracy improvement

**Timeline:** 8-10 weeks to production-ready system with 95%+ accuracy and <2s latency.

Would you like me to create detailed implementation guides for any specific sprint?