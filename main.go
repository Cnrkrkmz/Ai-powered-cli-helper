package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ─── Sabitler ────────────────────────────────────────────────────────────────

const (
	defaultOllamaURL  = "http://localhost:11434/api/generate"
	defaultModel      = "qwen2.5-coder:7b"
	historyFile       = "history.json"
	configFile        = "config.json"
	maxHistoryEntries = 200
)

// ─── Veri Yapıları ────────────────────────────────────────────────────────────

type Config struct {
	OllamaURL string `json:"ollama_url"`
	Model     string `json:"model"`
	NLModel   string `json:"nl_model"`  // NL modu için ayrı (opsiyonel, küçük model)
	Language  string `json:"language"`  // "tr" veya "en"
	MaxTokens int    `json:"max_tokens"`
}

type HistoryEntry struct {
	Timestamp   string `json:"timestamp"`
	Command     string `json:"command"`
	ErrorOutput string `json:"error_output"`
	Suggestion  string `json:"suggestion"`
	Executed    bool   `json:"executed"`
	Mode        string `json:"mode"` // "fix" veya "nl"
}

type OllamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// ToolMeta: bir knowledge dosyasının adı, yolu ve metadata satırlarını tutar
type ToolMeta struct {
	Name        string
	FilePath    string
	Description string // # TANIM: satırından
	Keywords    []string // # ANAHTAR KELİMELER: satırından (virgülle ayrılmış liste)
}

// ─── Ana Program ──────────────────────────────────────────────────────────────

func main() {
	args := os.Args[1:]

	if len(args) == 1 {
		switch args[0] {
		case "--history":
			showHistory()
			return
		case "--config":
			showConfig()
			return
		case "--help", "-h":
			showHelp()
			return
		case "--models":
			showAvailableModels()
			return
		}
	}

	if len(args) == 0 {
		showHelp()
		os.Exit(1)
	}

	cfg := loadConfig()

	if isNLQuery(args) {
		query := strings.Join(args, " ")
		handleNLMode(cfg, query)
		return
	}

	// ── FIX MODU ─────────────────────────────────────────────────────────────
	fullCommand := strings.Join(args, " ")

	userCmd := exec.Command(args[0], args[1:]...)
	errorOutput, err := userCmd.CombinedOutput()
	if err == nil {
		fmt.Print(string(errorOutput))
		os.Exit(0)
	}

	errorMessage := strings.TrimSpace(string(errorOutput))
	baseCommand := args[0]
	subCommand := extractSubCommand(args[1:])

	if cached := findInHistory(fullCommand, errorMessage); cached != "" {
		fmt.Printf("❌ Hata:\n%s\n\n", errorMessage)
		fmt.Printf("📦 Geçmiş öneri bulundu:\n")
		fmt.Printf("▶ Önerilen Komut: %s\n", cached)
		fmt.Printf("▶ Çalıştırayım mı? [e/h]: ")
		if readYesNo() {
			runCommand(cached)
		}
		return
	}

	toolContext := getSmartContext(baseCommand, subCommand)
	contextPrompt := ""
	if toolContext != "" {
		contextPrompt = fmt.Sprintf("--- REFERANS BİLGİ (%s %s için) ---\n%s\n\n",
			strings.ToUpper(baseCommand), strings.ToUpper(subCommand), toolContext)
	}

	prompt := buildFixPrompt(cfg.Language, contextPrompt, fullCommand, errorMessage)

	fmt.Printf("❌ Hata:\n%s\n\n", errorMessage)
	fmt.Printf("🤖 [%s] çözüm aranıyor...\n\n", cfg.Model)

	start := time.Now()
	fullResponse := askOllama(cfg.OllamaURL, cfg.Model, prompt, 150)
	elapsed := time.Since(start)

	suggestedCmd := cleanResponse(fullResponse)

	if suggestedCmd == "" {
		fmt.Println("⚠️  Model uygun bir komut üretemedi.")
		os.Exit(1)
	}

	if isDangerous(suggestedCmd) {
		fmt.Println("⛔ Güvenlik: Tehlikeli pattern tespit edildi. Engellendi.")
		os.Exit(1)
	}

	fmt.Printf("▶ Önerilen Komut: %s\n", suggestedCmd)
	fmt.Printf("▶ Süre: %.1fs | Model: %s\n", elapsed.Seconds(), cfg.Model)
	fmt.Printf("▶ Çalıştırayım mı? [e/h]: ")

	executed := false
	if readYesNo() {
		executed = true
		runCommand(suggestedCmd)
	} else {
		fmt.Println("İptal edildi.")
	}

	saveHistory(HistoryEntry{
		Timestamp:   time.Now().Format(time.RFC3339),
		Command:     fullCommand,
		ErrorOutput: errorMessage,
		Suggestion:  suggestedCmd,
		Executed:    executed,
		Mode:        "fix",
	})
}

// ─── NL MODU ─────────────────────────────────────────────────────────────────

func isNLQuery(args []string) bool {
	if len(args) == 0 {
		return false
	}
	if len(args) == 1 && strings.Contains(args[0], " ") {
		return true
	}
	firstArg := args[0]
	if strings.HasPrefix(firstArg, "-") {
		return false
	}
	_, err := exec.LookPath(firstArg)
	return err != nil
}

func handleNLMode(cfg Config, query string) {
	fmt.Printf("🧠 NL modu: \"%s\"\n\n", query)

	knowledgeFiles := listKnowledgeFiles()
	if len(knowledgeFiles) == 0 {
		fmt.Println("⚠️  ~/.term-ai/knowledge/ dizininde hiç dosya bulunamadı.")
		os.Exit(1)
	}

	// ── Adım 1: Araç tespiti — tamamen modelsiz, keyword matching ────────────
	fmt.Printf("🔍 Araç tespiti yapılıyor...\n")
	detectStart := time.Now()
	detectedTool := detectToolByKeywords(query, knowledgeFiles)
	detectElapsed := time.Since(detectStart)
	fmt.Printf("   → \"%s\" (%.0fms)\n\n", detectedTool, float64(detectElapsed.Microseconds())/1000)

	// ── Adım 2: İlgili knowledge dosyasını yükle ─────────────────────────────
	var toolContext string
	if detectedTool != "none" {
		if meta, ok := knowledgeFiles[detectedTool]; ok {
			toolContext = tryReadFile(meta.FilePath)
		}
	}
	if toolContext == "" {
		toolContext = gatherAllGeneralSections(knowledgeFiles)
	}

	// ── Adım 3: Komut üret — NLModel varsa onu kullan, yoksa Model ───────────
	nlModel := cfg.NLModel
	if nlModel == "" {
		nlModel = cfg.Model
	}
	fmt.Printf("🤖 [%s] komut üretiliyor...\n\n", nlModel)
	genStart := time.Now()
	prompt := buildNLPrompt(cfg.Language, toolContext, detectedTool, query)
	response := askOllama(cfg.OllamaURL, nlModel, prompt, 400)
	genElapsed := time.Since(genStart)

	suggestedCmd := cleanResponse(response)

	if suggestedCmd == "" {
		fmt.Println("⚠️  Model bir komut üretemedi. Knowledge dosyasını genişletmeyi dene.")
		os.Exit(1)
	}

	if isDangerous(suggestedCmd) {
		fmt.Println("⛔ Güvenlik: Tehlikeli pattern tespit edildi. Engellendi.")
		os.Exit(1)
	}

	totalElapsed := detectElapsed + genElapsed
	fmt.Printf("▶ Önerilen Komut: %s\n", suggestedCmd)
	fmt.Printf("▶ Toplam süre: %.1fs | Model: %s\n", totalElapsed.Seconds(), nlModel)
	fmt.Printf("▶ Çalıştırayım mı? [e/h]: ")

	executed := false
	if readYesNo() {
		executed = true
		runCommand(suggestedCmd)
	} else {
		fmt.Println("İptal edildi.")
	}

	saveHistory(HistoryEntry{
		Timestamp:  time.Now().Format(time.RFC3339),
		Command:    query,
		Suggestion: suggestedCmd,
		Executed:   executed,
		Mode:       "nl",
	})
}

// detectToolByKeywords: model çağrısı yapmadan, sorgu ile knowledge
// dosyalarının anahtar kelimelerini karşılaştırarak araç tespiti yapar.
//
// Puanlama:
//   +3  → araç adının kendisi sorguda geçiyor (en güçlü sinyal)
//   +2  → bir anahtar kelime tam kelime olarak eşleşiyor
//   +1  → bir anahtar kelime kısmi olarak eşleşiyor
//
// En yüksek puanlı araç döner; eşit puan varsa önce bulunan kazanır.
// Hiçbir araç en az 1 puan alamazsa "none" döner.
func detectToolByKeywords(query string, tools map[string]ToolMeta) string {
	queryLower := strings.ToLower(query)
	// Sorguyu kelimelere böl (noktalama temizle)
	queryWords := strings.FieldsFunc(queryLower, func(r rune) bool {
		return r == ' ' || r == '\'' || r == '"' || r == ',' || r == '.'
	})

	bestTool := "none"
	bestScore := 0

	for _, meta := range tools {
		score := 0
		nameLower := strings.ToLower(meta.Name)

		// Araç adının kendisi sorguda geçiyor mu?
		if strings.Contains(queryLower, nameLower) {
			score += 3
		}

		// Anahtar kelime eşleşmesi
		for _, kw := range meta.Keywords {
			kw = strings.TrimSpace(strings.ToLower(kw))
			if kw == "" {
				continue
			}
			// Tam kelime eşleşmesi
			for _, word := range queryWords {
				if word == kw {
					score += 2
					break
				}
			}
			// Kısmi eşleşme (sorgu keyword'ü içeriyor)
			if strings.Contains(queryLower, kw) {
				score += 1
			}
		}

		if score > bestScore {
			bestScore = score
			bestTool = meta.Name
		}
	}

	if bestScore == 0 {
		return "none"
	}
	return bestTool
}

// listKnowledgeFiles: ~/.term-ai/knowledge/*.txt dosyalarını okur,
// # TANIM ve # ANAHTAR KELİMELER satırlarını parse eder.
func listKnowledgeFiles() map[string]ToolMeta {
	homeDir, _ := os.UserHomeDir()
	dir := filepath.Join(homeDir, ".term-ai", "knowledge")

	entries, err := os.ReadDir(dir)
	if err != nil {
		return map[string]ToolMeta{}
	}

	files := make(map[string]ToolMeta)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".txt")
		path := filepath.Join(dir, e.Name())
		meta := ToolMeta{Name: name, FilePath: path}

		content := tryReadFile(path)
		for i, line := range strings.Split(content, "\n") {
			if i >= 10 {
				break
			}
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "# TANIM:") {
				meta.Description = strings.TrimSpace(strings.TrimPrefix(line, "# TANIM:"))
			} else if strings.HasPrefix(line, "# ANAHTAR KELİMELER:") {
				raw := strings.TrimSpace(strings.TrimPrefix(line, "# ANAHTAR KELİMELER:"))
				meta.Keywords = strings.Split(raw, ",")
			}
		}

		files[name] = meta
	}
	return files
}

func gatherAllGeneralSections(files map[string]ToolMeta) string {
	var parts []string
	for name, meta := range files {
		content := tryReadFile(meta.FilePath)
		if content == "" {
			continue
		}
		general := extractSection(content, "general")
		if general != "" {
			parts = append(parts, fmt.Sprintf("=== %s ===\n%s", strings.ToUpper(name), general))
		}
	}
	return strings.Join(parts, "\n\n")
}

// ─── Prompt Üreticiler ────────────────────────────────────────────────────────

func buildFixPrompt(lang, contextPrompt, command, errorMsg string) string {
	if lang == "en" {
		return fmt.Sprintf(`You are an expert system engineer. Fix the broken terminal command.
Rules:
1. Output ONLY the corrected command. No explanation, no markdown, no backticks.
2. If reference info is provided, follow those rules strictly.

%s--- BROKEN COMMAND ---
%s

--- ERROR OUTPUT ---
%s

CORRECTED COMMAND:`, contextPrompt, command, errorMsg)
	}

	return fmt.Sprintf(`Sen uzman bir sistem mühendisisin. Hatalı terminal komutunu düzelt.
Kurallar:
1. Sadece düzeltilmiş komutu yaz. Açıklama, markdown veya backtick KULLANMA.
2. Referans bilgi verilmişse o kurallara kesinlikle uy.

%s--- HATALI KOMUT ---
%s

--- HATA ÇIKTISI ---
%s

DÜZELTİLMİŞ KOMUT:`, contextPrompt, command, errorMsg)
}

func buildNLPrompt(lang, toolContext, detectedTool string, query string) string {
	contextBlock := ""
	if toolContext != "" {
		label := strings.ToUpper(detectedTool)
		if detectedTool == "none" || detectedTool == "" {
			label = "GENEL"
		}
		contextBlock = fmt.Sprintf("--- REFERANS BİLGİ (%s) ---\n%s\n\n", label, toolContext)
	}

	if lang == "en" {
		return fmt.Sprintf(`You are an expert system engineer.
Generate the correct terminal command for the given task.
Rules:
1. Output ONLY the command. No explanation, no markdown, no backticks.
2. Use the reference info strictly. Do not invent flags that are not listed.
3. For unknown values (passwords, domains, IPs) use <PLACEHOLDER> format.

%s--- TASK ---
%s

COMMAND:`, contextBlock, query)
	}

	return fmt.Sprintf(`Sen uzman bir sistem mühendisisin.
Verilen görev için doğru terminal komutunu üret.
Kurallar:
1. Sadece komutu yaz. Açıklama, markdown veya backtick KULLANMA.
2. Referans bilgideki flag ve kurallara kesinlikle uy. Listede olmayan flag uydurma.
3. Bilinmesi gereken değerler (şifre, domain, IP vb.) için <PLACEHOLDER> formatını kullan.

%s--- GÖREV ---
%s

KOMUT:`, contextBlock, query)
}

// ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

func extractSubCommand(args []string) string {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			return strings.ToLower(arg)
		}
	}
	return "general"
}

func cleanResponse(raw string) string {
	s := strings.TrimSpace(raw)
	s = regexp.MustCompile("(?s)```[a-z]*\n?(.*?)```").ReplaceAllString(s, "$1")
	s = strings.Trim(s, "`")
	lines := strings.Split(s, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") && !strings.HasPrefix(line, "//") {
			return line
		}
	}
	return ""
}

func isDangerous(cmd string) bool {
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)rm\s+-[a-z]*r[a-z]*f`),
		regexp.MustCompile(`(?i)>\s*/dev/[sh]`),
		regexp.MustCompile(`(?i)\bmkfs\b`),
		regexp.MustCompile(`(?i)mv\s+/\s+`),
		regexp.MustCompile(`(?i)chmod\s+-R\s+777\s+/`),
		regexp.MustCompile(`(?i)dd\s+.*of=/dev/[sh]d`),
		regexp.MustCompile(`(?i):\(\)\{.*\|.*&\}`),
	}
	for _, p := range patterns {
		if p.MatchString(cmd) {
			return true
		}
	}
	return false
}

func readYesNo() bool {
	reader := bufio.NewReader(os.Stdin)
	answer, _ := reader.ReadString('\n')
	answer = strings.TrimSpace(strings.ToLower(answer))
	return answer == "e" || answer == "evet" || answer == "y" || answer == "yes"
}

func runCommand(cmd string) {
	fmt.Println("▶▶ Çalıştırılıyor...")
	fmt.Println(strings.Repeat("─", 50))
	execCmd := exec.Command("sh", "-c", cmd)
	execCmd.Stdout = os.Stdout
	execCmd.Stderr = os.Stderr
	execCmd.Stdin = os.Stdin
	if err := execCmd.Run(); err != nil {
		fmt.Printf("\n▶▶▶ Komut hata döndürdü: %v\n", err)
	}
}

// ─── Ollama ───────────────────────────────────────────────────────────────────

func askOllama(ollamaURL, model, prompt string, numPredict int) string {
	reqBody := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.1,
			"num_predict": numPredict,
			"stop":        []string{"\n\n", "---"},
		},
	}

	jsonData, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 90 * time.Second}

	resp, err := client.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("❌ Ollama'ya ulaşılamadı (%s)\n", ollamaURL)
		fmt.Println("   → 'ollama serve' çalışıyor mu?")
		fmt.Printf("   → Model kurulu mu? 'ollama pull %s'\n", model)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var chunk OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&chunk); err != nil {
		return ""
	}
	return chunk.Response
}

// ─── Context (Akıllı Grep) ───────────────────────────────────────────────────

func getSmartContext(baseCmd, subCmd string) string {
	homeDir, _ := os.UserHomeDir()
	dir := filepath.Join(homeDir, ".term-ai", "knowledge")
	content := tryReadFile(filepath.Join(dir, baseCmd+".txt"))
	if content == "" {
		return ""
	}
	return extractSection(content, subCmd)
}

func tryReadFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

func extractSection(fileContent, subCmd string) string {
	headerRe := regexp.MustCompile(`(?m)^\[([^\]]+)\]\s*$`)
	headers := headerRe.FindAllStringSubmatchIndex(fileContent, -1)

	if len(headers) == 0 {
		return strings.TrimSpace(fileContent)
	}

	sections := make(map[string]string)
	for i, loc := range headers {
		headerName := strings.ToLower(strings.TrimSpace(fileContent[loc[2]:loc[3]]))
		start := loc[1]
		end := len(fileContent)
		if i+1 < len(headers) {
			end = headers[i+1][0]
		}
		sections[headerName] = strings.TrimSpace(fileContent[start:end])
	}

	if val, ok := sections[subCmd]; ok {
		return val
	}
	for key, val := range sections {
		if strings.Contains(key, subCmd) || strings.Contains(subCmd, key) {
			return val
		}
	}
	for _, fallback := range []string{"general", "genel"} {
		if val, ok := sections[fallback]; ok {
			return val
		}
	}
	return ""
}

// ─── Config ───────────────────────────────────────────────────────────────────

func termAIDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".term-ai")
}

func loadConfig() Config {
	defaults := Config{
		OllamaURL: defaultOllamaURL,
		Model:     defaultModel,
		NLModel:   "",
		Language:  "tr",
		MaxTokens: 150,
	}

	data, err := os.ReadFile(filepath.Join(termAIDir(), configFile))
	if err != nil {
		return defaults
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return defaults
	}

	if cfg.OllamaURL == "" { cfg.OllamaURL = defaults.OllamaURL }
	if cfg.Model == "" { cfg.Model = defaults.Model }
	if cfg.Language == "" { cfg.Language = defaults.Language }
	if cfg.MaxTokens == 0 { cfg.MaxTokens = defaults.MaxTokens }

	return cfg
}

func showConfig() {
	cfg := loadConfig()
	fmt.Printf("📋 Config: %s\n\n", filepath.Join(termAIDir(), configFile))
	fmt.Printf("  model:      %s  (fix modu)\n", cfg.Model)
	nlModel := cfg.NLModel
	if nlModel == "" {
		nlModel = cfg.Model + "  (nl_model ayarlanmamış, model kullanılıyor)"
	}
	fmt.Printf("  nl_model:   %s  (NL modu komut üretimi)\n", nlModel)
	fmt.Printf("  ollama_url: %s\n", cfg.OllamaURL)
	fmt.Printf("  language:   %s\n", cfg.Language)
	fmt.Printf("\nDeğiştirmek için dosyayı düzenleyin.\n")
}

// ─── Geçmiş ───────────────────────────────────────────────────────────────────

func historyPath() string {
	return filepath.Join(termAIDir(), historyFile)
}

func loadHistory() []HistoryEntry {
	data, err := os.ReadFile(historyPath())
	if err != nil {
		return []HistoryEntry{}
	}
	var entries []HistoryEntry
	json.Unmarshal(data, &entries)
	return entries
}

func saveHistory(entry HistoryEntry) {
	os.MkdirAll(termAIDir(), 0755)
	entries := loadHistory()
	entries = append(entries, entry)
	if len(entries) > maxHistoryEntries {
		entries = entries[len(entries)-maxHistoryEntries:]
	}
	data, _ := json.MarshalIndent(entries, "", "  ")
	os.WriteFile(historyPath(), data, 0644)
}

func findInHistory(command, errorMsg string) string {
	entries := loadHistory()
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		if e.Command == command && e.ErrorOutput == errorMsg && e.Suggestion != "" {
			return e.Suggestion
		}
	}
	return ""
}

func showHistory() {
	entries := loadHistory()
	if len(entries) == 0 {
		fmt.Println("📭 Henüz geçmiş yok.")
		return
	}
	start := 0
	if len(entries) > 20 {
		start = len(entries) - 20
	}
	fmt.Printf("📜 Son %d kayıt:\n\n", len(entries[start:]))
	for _, e := range entries[start:] {
		status := "⬜"
		if e.Executed {
			status = "✅"
		}
		modeIcon := "🔧"
		if e.Mode == "nl" {
			modeIcon = "🧠"
		}
		t, _ := time.Parse(time.RFC3339, e.Timestamp)
		fmt.Printf("%s %s [%s] %s\n   → %s\n\n",
			status, modeIcon, t.Format("02 Jan 15:04"), e.Command, e.Suggestion)
	}
}

// ─── Yardım & Model Listesi ───────────────────────────────────────────────────

func showHelp() {
	fmt.Println(`🤖 term-ai — Terminal için AI Asistan

Kullanım:
  term-ai <komut> [argümanlar]     🔧 FIX: Komutu çalıştır, hata varsa düzelt
  term-ai "<doğal dil sorgusu>"    🧠 NL:  Komut bilmeden doğal dille sor
  term-ai --history                Son önerileri göster
  term-ai --config                 Mevcut ayarları göster
  term-ai --models                 Kurulu Ollama modellerini listele
  term-ai --help                   Bu yardım mesajı

FIX modu örnekleri:
  term-ai git psuh origin main
  term-ai kubectl get pods -n producton
  term-ai docker-compose up -d --buld

NL modu örnekleri:
  term-ai "instana backendi production modda ayağa kaldır"
  term-ai "ahmet@sirket.com kullanıcısının 2fa'sını sıfırla"
  term-ai instana object dizinini değiştir   ← tırnaksız da çalışır

NL modu için küçük model ayarı (~/.term-ai/config.json):
  "nl_model": "gemma3:1b"   ← araç tespiti modelsiz, komut üretimi bu modelle

Ayarlar: ~/.term-ai/config.json
Bilgi deposu: ~/.term-ai/knowledge/<komut>.txt`)
}

func showAvailableModels() {
	fmt.Print("📦 Kurulu Ollama modelleri:\n\n")
	cmd := exec.Command("ollama", "list")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Println("❌ 'ollama list' çalıştırılamadı. Ollama kurulu ve çalışıyor mu?")
	}
	fmt.Printf("\nfix modu için (config: \"model\"):\n")
	fmt.Printf("  %-30s %s\n", "qwen2.5-coder:7b", "(varsayılan, ~5GB)")
	fmt.Printf("  %-30s %s\n", "qwen2.5-coder:14b", "(daha güçlü, ~9GB)")
	fmt.Printf("\nNL modu için (config: \"nl_model\") — hız öncelikli:\n")
	fmt.Printf("  %-30s %s\n", "gemma3:1b", "(en hızlı, ~800MB) ← önerilir")
	fmt.Printf("  %-30s %s\n", "qwen2.5:0.5b", "(çok küçük, ~400MB)")
	fmt.Printf("  %-30s %s\n", "gemma3:4b", "(dengeli, ~3GB)")
	fmt.Printf("\nKurmak için: ollama pull <model-adı>\n")
}