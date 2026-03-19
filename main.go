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
	Language  string `json:"language"` // "tr" veya "en"
	MaxTokens int    `json:"max_tokens"`
}

type HistoryEntry struct {
	Timestamp   string `json:"timestamp"`
	Command     string `json:"command"`
	ErrorOutput string `json:"error_output"`
	Suggestion  string `json:"suggestion"`
	Executed    bool   `json:"executed"`
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

// ─── Ana Program ──────────────────────────────────────────────────────────────

func main() {
	args := os.Args[1:]

	// Özel komutlar
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
	fullCommand := strings.Join(args, " ")

	// ── 1. Komutu çalıştır ──
	userCmd := exec.Command(args[0], args[1:]...)
	errorOutput, err := userCmd.CombinedOutput()
	if err == nil {
		// Komut başarılı, çıktıyı olduğu gibi göster
		fmt.Print(string(errorOutput))
		os.Exit(0)
	}

	errorMessage := strings.TrimSpace(string(errorOutput))
	baseCommand := args[0]

	subCommand := extractSubCommand(args[1:])

	// ── 2. Aynı hatayı daha önce çözdük mü? ──
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

	// ── 3. Context'i hazırla ──
	toolContext := getSmartContext(baseCommand, subCommand)
	contextPrompt := ""
	if toolContext != "" {
		contextPrompt = fmt.Sprintf("--- REFERANS BİLGİ (%s %s için) ---\n%s\n\n",
			strings.ToUpper(baseCommand), strings.ToUpper(subCommand), toolContext)
	}

	// ── 4. Prompt ──
	prompt := buildPrompt(cfg.Language, contextPrompt, fullCommand, errorMessage)

	fmt.Printf("❌ Hata:\n%s\n\n", errorMessage)
	fmt.Printf("🤖 [%s] çözüm aranıyor...\n\n", cfg.Model)

	// ── 5. Ollama'ya sor ──
	start := time.Now()
	fullResponse := askOllama(cfg, prompt)
	elapsed := time.Since(start)

	suggestedCmd := cleanResponse(fullResponse)

	if suggestedCmd == "" {
		fmt.Println("⚠️  Model uygun bir komut üretemedi.")
		os.Exit(1)
	}

	// ── 6. Güvenlik kontrolü ──
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

	// ── 7. Geçmişe kaydet ──
	saveHistory(HistoryEntry{
		Timestamp:   time.Now().Format(time.RFC3339),
		Command:     fullCommand,
		ErrorOutput: errorMessage,
		Suggestion:  suggestedCmd,
		Executed:    executed,
	})
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

func buildPrompt(lang, contextPrompt, command, errorMsg string) string {
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

	// Türkçe (varsayılan)
	return fmt.Sprintf(`Sen uzman bir sistem mühendisisin. Hatalı terminal komutunu düzelt.
Kurallar:
1. Sadece düzeltilmiş komutu yaz. Açıklama, markdown veya backtick KULLANMA.
2. Referans bilgi verilmişse o kurallara kesinlikle uy.

%s--- HATALI KOMUT ---
%s

--- HATA ÇIKTISI ---
%s

DÜZELTILMIŞ KOMUT:`, contextPrompt, command, errorMsg)
}

func cleanResponse(raw string) string {
	s := strings.TrimSpace(raw)
	// Backtick bloklarını temizle
	s = regexp.MustCompile("(?s)```[a-z]*\n?(.*?)```").ReplaceAllString(s, "$1")
	s = strings.Trim(s, "`")
	// Sadece ilk satırı al (model açıklama eklediyse)
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
	dangerousPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)rm\s+-[a-z]*r[a-z]*f`),
		regexp.MustCompile(`(?i)>\s*/dev/[sh]`),
		regexp.MustCompile(`(?i)\bmkfs\b`),
		regexp.MustCompile(`(?i)mv\s+/\s+`),
		regexp.MustCompile(`(?i)chmod\s+-R\s+777\s+/`),
		regexp.MustCompile(`(?i)dd\s+.*of=/dev/[sh]d`),
		regexp.MustCompile(`(?i):\(\)\{.*\|.*&\}`), // fork bomb
	}
	for _, p := range dangerousPatterns {
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

func askOllama(cfg Config, prompt string) string {
	reqBody := OllamaRequest{
		Model:  cfg.Model,
		Prompt: prompt,
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.1, // Düşük: tutarlı, öngörülebilir çıktı
			"num_predict": 150, // Komutlar kısa olur, fazlasına gerek yok
			"stop":        []string{"\n\n", "---"},
		},
	}

	jsonData, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 60 * time.Second}

	resp, err := client.Post(cfg.OllamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("❌ Ollama'ya ulaşılamadı (%s)\n", cfg.OllamaURL)
		fmt.Println("   → 'ollama serve' çalışıyor mu?")
		fmt.Printf("   → Model kurulu mu? 'ollama pull %s'\n", cfg.Model)
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
	termAIDir := filepath.Join(homeDir, ".term-ai", "knowledge")

	// Önce tam eşleşme: kubectl.txt
	content := tryReadFile(filepath.Join(termAIDir, baseCmd+".txt"))
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
	// Dosyayı [başlık] bloklarına göre ayır
	headerRe := regexp.MustCompile(`(?m)^\[([^\]]+)\]\s*$`)
	headers := headerRe.FindAllStringSubmatchIndex(fileContent, -1)

	if len(headers) == 0 {
		// Başlık yapısı yok → tüm dosyayı döndür (küçük dosyalar için)
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

	// 1. Tam eşleşme
	if val, ok := sections[subCmd]; ok {
		return val
	}

	// 2. Kısmi eşleşme (örn: subCmd="upgrade", header="upgrade-cluster")
	for key, val := range sections {
		if strings.Contains(key, subCmd) || strings.Contains(subCmd, key) {
			return val
		}
	}

	// 3. [general] veya [genel] varsa onu döndür
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
		Language:  "tr",
		MaxTokens: 150,
	}

	path := filepath.Join(termAIDir(), configFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return defaults
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return defaults
	}

	// Eksik alanları doldur
	if cfg.OllamaURL == "" {
		cfg.OllamaURL = defaults.OllamaURL
	}
	if cfg.Model == "" {
		cfg.Model = defaults.Model
	}
	if cfg.Language == "" {
		cfg.Language = defaults.Language
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaults.MaxTokens
	}

	return cfg
}

func showConfig() {
	cfg := loadConfig()
	path := filepath.Join(termAIDir(), configFile)
	fmt.Printf("📋 Config: %s\n\n", path)
	fmt.Printf("  model:      %s\n", cfg.Model)
	fmt.Printf("  ollama_url: %s\n", cfg.OllamaURL)
	fmt.Printf("  language:   %s\n", cfg.Language)
	fmt.Printf("  max_tokens: %d\n", cfg.MaxTokens)
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
	// Maksimum sayıyı aşarsa en eskiyi sil
	if len(entries) > maxHistoryEntries {
		entries = entries[len(entries)-maxHistoryEntries:]
	}
	data, _ := json.MarshalIndent(entries, "", "  ")
	os.WriteFile(historyPath(), data, 0644)
}

func findInHistory(command, errorMsg string) string {
	entries := loadHistory()
	// Sondan başa ara (en yeni önce)
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
	fmt.Printf("📜 Son %d kayıt:\n\n", len(entries))
	// Son 20'yi göster
	start := 0
	if len(entries) > 20 {
		start = len(entries) - 20
	}
	for _, e := range entries[start:] {
		status := "⬜"
		if e.Executed {
			status = "✅"
		}
		t, _ := time.Parse(time.RFC3339, e.Timestamp)
		fmt.Printf("%s [%s] %s\n   → %s\n\n",
			status, t.Format("02 Jan 15:04"), e.Command, e.Suggestion)
	}
}

// ─── Yardım & Model Listesi ───────────────────────────────────────────────────

func showHelp() {
	fmt.Println(`🤖 term-ai — Terminal için AI Asistan

Kullanım:
  term-ai <komut> [argümanlar]   Komutu çalıştır, hata varsa düzelt
  term-ai --history              Son önerileri göster
  term-ai --config               Mevcut ayarları göster
  term-ai --models               Kurulu Ollama modellerini listele
  term-ai --help                 Bu yardım mesajı

Örnekler:
  term-ai git psuh origin main
  term-ai kubectl get pods -n producton
  term-ai docker-compose up -d --buld

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
	fmt.Printf("\nModel değiştirmek için: ~/.term-ai/config.json → \"model\" alanını düzenle\n")
	fmt.Printf("Önerilen modeller:\n")
	fmt.Printf("  %-30s %s\n", "qwen2.5-coder:7b", "(varsayılan, ~5GB, iyi denge)")
	fmt.Printf("  %-30s %s\n", "qwen2.5-coder:14b", "(daha güçlü, ~9GB)")
	fmt.Printf("  %-30s %s\n", "gemma3:4b", "(hızlı, ~3GB, çok dilli)")
	fmt.Printf("  %-30s %s\n", "deepseek-coder-v2:16b", "(en güçlü, ~10GB)")
}
