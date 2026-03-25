package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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
	trainingDataFile  = "training_data.json"
	maxHistoryEntries = 200
)

// ─── Veri Yapıları ────────────────────────────────────────────────────────────

type Config struct {
	OllamaURL     string `json:"ollama_url"`
	Model         string `json:"model"`
	NLModel       string `json:"nl_model"`
	Language      string `json:"language"`
	MaxTokens     int    `json:"max_tokens"`
	CloudProvider string `json:"cloud_provider"` // "anthropic" | "gemini" | "openai"
	CloudAPIKey   string `json:"cloud_api_key"`
	CloudModel    string `json:"cloud_model"`
}

type HistoryEntry struct {
	Timestamp   string `json:"timestamp"`
	Command     string `json:"command"`
	ErrorOutput string `json:"error_output"`
	Suggestion  string `json:"suggestion"`
	Executed    bool   `json:"executed"`
	Mode        string `json:"mode"` // "fix" | "nl"
	Backend     string `json:"backend"` // "local" | "cloud"
}

// TrainingExample: Fine-tuning için kullanılacak veri formatı
type TrainingExample struct {
	Instruction string `json:"instruction"` // Kullanıcının doğal dil sorgusu
	Output      string `json:"output"`      // Onaylanmış komut
	Tool        string `json:"tool"`        // Tespit edilen araç (git, stanctl, etc.)
	Timestamp   string `json:"timestamp"`   // Ne zaman eklendi
	Source      string `json:"source"`      // "user_approved" | "manual" | "synthetic"
}

// ToolMeta: bir araç klasörünün _meta.txt'sinden okunan bilgiler
type ToolMeta struct {
	Name        string
	Dir         string
	Description string
	Keywords    []string
	SubFiles    map[string]string
}

// Ollama tipleri
type OllamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaStreamChunk struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// Anthropic tipleri
type AnthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Stream    bool               `json:"stream"`
	Messages  []AnthropicMessage `json:"messages"`
}

type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAI / Gemini (OpenAI compat) tipleri
type OpenAIRequest struct {
	Model    string          `json:"model"`
	Stream   bool            `json:"stream"`
	Messages []OpenAIMessage `json:"messages"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
		case "--index":
			runIndex(loadConfig(), true)
			return
		case "--training-stats":
			showTrainingStats()
			return
		}
	}

	if len(args) == 0 {
		showHelp()
		os.Exit(1)
	}

	cfg := loadConfig()

	// ── Flag'leri parse et ───────────────────────────────────────────────────
	ignoreHistory := false
	useCloud := false
	useChat := false
	filteredArgs := args[:0]
	for _, a := range args {
		switch a {
		case "--no-history", "--ignore-hist":
			ignoreHistory = true
		case "--cloud":
			useCloud = true
		case "--chat":
			useChat = true
		default:
			filteredArgs = append(filteredArgs, a)
		}
	}
	args = filteredArgs

	if len(args) == 0 {
		showHelp()
		os.Exit(1)
	}

	if useChat {
		query := strings.Join(args, " ")
		handleChatMode(cfg, query, ignoreHistory, useCloud)
		return
	}

	if isNLQuery(args) {
		query := strings.Join(args, " ")
		handleNLMode(cfg, query, ignoreHistory, useCloud)
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

	// Geçmiş kontrolü
	if !ignoreHistory {
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
	}

	// Maskeleme
	safeErrorMessage := maskSensitiveData(errorMessage)

	toolContext := getSmartContext(baseCommand, subCommand)
	contextPrompt := ""
	if toolContext != "" {
		contextPrompt = fmt.Sprintf("--- REFERANS BİLGİ (%s %s için) ---\n%s\n\n",
			strings.ToUpper(baseCommand), strings.ToUpper(subCommand), toolContext)
	}

	prompt := buildFixPrompt(cfg.Language, contextPrompt, fullCommand, safeErrorMessage)

	backend := "local"
	if useCloud {
		backend = "cloud"
	}

	fmt.Printf("❌ Hata:\n%s\n\n", errorMessage)
	if safeErrorMessage != errorMessage {
		fmt.Printf("🔒 Hassas veri maskelendi\n\n")
	}

	model := resolveModel(cfg, useCloud, false)
	fmt.Printf("🤖 [%s/%s] çözüm aranıyor...\n\n", backend, model)

	start := time.Now()
	suggestedCmd := askLLM(cfg, prompt, 150, useCloud)
	elapsed := time.Since(start)

	suggestedCmd = cleanResponse(suggestedCmd)

	if suggestedCmd == "" {
		fmt.Println("\n⚠️  Model uygun bir komut üretemedi.")
		os.Exit(1)
	}

	if isDangerous(suggestedCmd) {
		fmt.Println("\n⛔ Güvenlik: Tehlikeli pattern tespit edildi. Engellendi.")
		os.Exit(1)
	}

	fmt.Printf("\n▶ Önerilen Komut: %s\n", suggestedCmd)
	fmt.Printf("▶ Süre: %.1fs | %s/%s\n", elapsed.Seconds(), backend, model)
	fmt.Printf("▶ Çalıştırayım mı? [e/h]: ")

	executed := false
	if readYesNo() {
		executed = true
		runCommand(suggestedCmd)
	} else {
		fmt.Println("İptal edildi.")
	}

	if !ignoreHistory {
		saveHistory(HistoryEntry{
			Timestamp:   time.Now().Format(time.RFC3339),
			Command:     fullCommand,
			ErrorOutput: safeErrorMessage,
			Suggestion:  suggestedCmd,
			Executed:    executed,
			Mode:        "fix",
			Backend:     backend,
		})
	}
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

func handleNLMode(cfg Config, query string, ignoreHistory bool, useCloud bool) {
	fmt.Printf("🧠 NL modu: \"%s\"\n\n", query)

	tools := listTools()
	if len(tools) == 0 {
		fmt.Println("⚠️  ~/.term-ai/knowledge/ dizininde hiç araç klasörü bulunamadı.")
		os.Exit(1)
	}

	// ── Adım 1: Araç + dosya tespiti (vektör önce, keyword fallback) ──────────
	fmt.Printf("🔍 Arama yapılıyor...\n")
	t0 := time.Now()

	detectedTool := "none"
	subFile := "general"
	searchMethod := "keyword"

	// Önce vektör arama dene (index mevcutsa ve güncelsse)
	if !IsIndexStale(cfg.OllamaURL) {
		result := SearchSimilar(cfg.OllamaURL, query)
		if result.Tool != "none" {
			detectedTool = result.Tool
			subFile = result.SubFile
			searchMethod = fmt.Sprintf("vektör (%.2f)", result.Score)
		}
	}

	// Vektör başarısızsa keyword'e düş
	if detectedTool == "none" {
		detectedTool = detectToolByKeywords(query, tools)
		if detectedTool != "none" {
			meta := tools[detectedTool]
			subFile = detectSubFileByKeywords(query, meta)
		}
	}

	t1 := time.Since(t0)
	if detectedTool != "none" {
		fmt.Printf("   → %s/%s [%s] (%.0fms)\n\n",
			detectedTool, subFile, searchMethod, float64(t1.Microseconds())/1000)
	} else {
		fmt.Printf("   → eşleşme bulunamadı, genel context kullanılıyor\n\n")
	}

	// ── Adım 2: Context yükle ─────────────────────────────────────────────────
	var toolContext string
	if detectedTool != "none" {
		meta := tools[detectedTool]
		toolContext = loadSubFileContext(meta, subFile)
	}
	if toolContext == "" {
		toolContext = gatherAllGeneralSections(tools)
	}

	// ── Adım 3: Komut üret ───────────────────────────────────────────────────
	backend := "local"
	if useCloud {
		backend = "cloud"
	}
	model := resolveModel(cfg, useCloud, true)
	fmt.Printf("🤖 [%s/%s] komut üretiliyor...\n\n", backend, model)

	t2 := time.Now()
	prompt := buildNLPrompt(cfg.Language, toolContext, detectedTool, query)
	suggestedCmd := askLLM(cfg, prompt, 400, useCloud)
	t3 := time.Since(t2)

	suggestedCmd = cleanResponse(suggestedCmd)

	if suggestedCmd == "" {
		fmt.Println("\n⚠️  Model bir komut üretemedi.")
		os.Exit(1)
	}

	if isDangerous(suggestedCmd) {
		fmt.Println("\n⛔ Güvenlik: Tehlikeli pattern tespit edildi. Engellendi.")
		os.Exit(1)
	}

	fmt.Printf("\n▶ Önerilen Komut: %s\n", suggestedCmd)
	fmt.Printf("▶ Toplam süre: %.1fs | %s/%s\n", (t1 + t3).Seconds(), backend, model)
	fmt.Printf("▶ Çalıştırayım mı? [e/h]: ")

	executed := false
	if readYesNo() {
		executed = true
		runCommand(suggestedCmd)
		
		// Başarılı komutları training data'ya ekle
		if detectedTool == "none" {
			detectedTool = "general"
		}
		addApprovedCommand(query, suggestedCmd, detectedTool)
		fmt.Println("✅ Komut training data'ya eklendi")
	} else {
		fmt.Println("İptal edildi.")
	}

	if !ignoreHistory {
		saveHistory(HistoryEntry{
			Timestamp:  time.Now().Format(time.RFC3339),
			Command:    query,
			Suggestion: suggestedCmd,
			Executed:   executed,
			Mode:       "nl",
			Backend:    backend,
		})
	}
}

// ─── CHAT MODU ────────────────────────────────────────────────────────────────

// handleChatMode: --chat flag’i ile serbest konuşma modu.
// Terminal komutu üretme zorunluluğu yok, model istediği gibi cevap verebilir.
func handleChatMode(cfg Config, query string, ignoreHistory bool, useCloud bool) {
	backend := "local"
	if useCloud {
		backend = "cloud"
	}
	model := resolveModel(cfg, useCloud, true)
	fmt.Printf("💬 Sohbet modu [%s/%s]\n\n", backend, model)

	safeQuery := maskSensitiveData(query)
	if safeQuery != query {
		fmt.Printf("🔒 Hassas veri maskelendi\n\n")
	}

	prompt := buildChatPrompt(cfg.Language, safeQuery)

	start := time.Now()
	response := askLLM(cfg, prompt, 1000, useCloud)
	elapsed := time.Since(start)

	response = strings.TrimSpace(response)
	if response == "" {
		fmt.Println("\n⚠️  Model cevap üretemedi.")
		os.Exit(1)
	}

	fmt.Printf("\n\n▶ Süre: %.1fs | %s/%s\n", elapsed.Seconds(), backend, model)

	if !ignoreHistory {
		saveHistory(HistoryEntry{
			Timestamp:  time.Now().Format(time.RFC3339),
			Command:    query,
			Suggestion: response,
			Executed:   false,
			Mode:       "chat",
			Backend:    backend,
		})
	}
}

func buildChatPrompt(lang, query string) string {
	if lang == "en" {
		return fmt.Sprintf(`You are a helpful terminal assistant. Answer the following question clearly and concisely.
Do NOT force a terminal command if the question is conversational.

Question: %s`, query)
	}
	return fmt.Sprintf(`Sen yardımcı bir terminal asistanısın. Aşağıdaki soruyu açık ve net şekilde yanıtla.
Eğer soru genel veya sohbet amaçlıysa terminal komutu vermeye zorlanma, düz metin olarak cevap ver.

Soru: %s`, query)
}

// ─── LLM Yönlendirici ─────────────────────────────────────────────────────────

// resolveModel: hangi modelin kullanıldığını döner (görüntüleme için)
func resolveModel(cfg Config, useCloud bool, isNL bool) string {
	if useCloud {
		if cfg.CloudModel != "" {
			return cfg.CloudModel
		}
		switch cfg.CloudProvider {
		case "anthropic":
			return "claude-haiku-4-5"
		case "gemini":
			return "gemini-2.0-flash"
		case "openai":
			return "gpt-4o-mini"
		default:
			return "cloud-model"
		}
	}
	if isNL && cfg.NLModel != "" {
		return cfg.NLModel
	}
	return cfg.Model
}

// askLLM: cloud=true ise cloud API'ye, false ise Ollama'ya gönderir.
// Her iki durumda da streaming kullanır.
func askLLM(cfg Config, prompt string, maxTokens int, useCloud bool) string {
	if useCloud {
		if cfg.CloudAPIKey == "" {
			fmt.Println("❌ Cloud modu için config.json'a cloud_api_key eklemelisin.")
			os.Exit(1)
		}
		return askCloud(cfg, prompt, maxTokens)
	}

	model := cfg.Model
	if cfg.NLModel != "" {
		model = cfg.NLModel
	}
	return askOllamaStream(cfg.OllamaURL, model, prompt, maxTokens)
}

// ─── Ollama (Streaming) ───────────────────────────────────────────────────────

func askOllamaStream(ollamaURL, model, prompt string, numPredict int) string {
	reqBody := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: true,
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

	var fullResponse strings.Builder
	decoder := json.NewDecoder(resp.Body)

	for {
		var chunk OllamaStreamChunk
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			break
		}
		if chunk.Response != "" {
			fmt.Print(chunk.Response)
			fullResponse.WriteString(chunk.Response)
		}
		if chunk.Done {
			break
		}
	}

	return fullResponse.String()
}

// ─── Cloud API (Streaming) ────────────────────────────────────────────────────

func askCloud(cfg Config, prompt string, maxTokens int) string {
	switch cfg.CloudProvider {
	case "anthropic":
		return askAnthropic(cfg, prompt, maxTokens)
	case "openai":
		return askOpenAICompat(cfg, prompt, maxTokens,
			"https://api.openai.com/v1/chat/completions",
			"Bearer "+cfg.CloudAPIKey)
	case "gemini":
		model := cfg.CloudModel
		if model == "" {
			model = "gemini-2.0-flash"
		}
		url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?model=%s", model)
		return askOpenAICompat(cfg, prompt, maxTokens, url, "Bearer "+cfg.CloudAPIKey)
	default:
		fmt.Printf("❌ Bilinmeyen cloud provider: %s (anthropic | gemini | openai)\n", cfg.CloudProvider)
		os.Exit(1)
	}
	return ""
}

// askAnthropic: Anthropic Messages API, streaming
func askAnthropic(cfg Config, prompt string, maxTokens int) string {
	model := cfg.CloudModel
	if model == "" {
		model = "claude-haiku-4-5"
	}

	reqBody := AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Stream:    true,
		Messages:  []AnthropicMessage{{Role: "user", Content: prompt}},
	}

	jsonData, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 60 * time.Second}

	req, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", cfg.CloudAPIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("❌ Anthropic API hatası: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("❌ Anthropic API %d: %s\n", resp.StatusCode, string(body))
		os.Exit(1)
	}

	// SSE stream parse
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event map[string]interface{}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		// content_block_delta eventi
		if event["type"] == "content_block_delta" {
			if delta, ok := event["delta"].(map[string]interface{}); ok {
				if text, ok := delta["text"].(string); ok {
					fmt.Print(text)
					fullResponse.WriteString(text)
				}
			}
		}
	}

	return fullResponse.String()
}

// askOpenAICompat: OpenAI ve Gemini (OpenAI uyumlu), streaming
func askOpenAICompat(cfg Config, prompt string, maxTokens int, apiURL, authHeader string) string {
	model := cfg.CloudModel
	if model == "" {
		if cfg.CloudProvider == "gemini" {
			model = "gemini-2.0-flash"
		} else {
			model = "gpt-4o-mini"
		}
	}

	reqBody := OpenAIRequest{
		Model:    model,
		Stream:   true,
		Messages: []OpenAIMessage{{Role: "user", Content: prompt}},
	}

	jsonData, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 60 * time.Second}

	req, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", authHeader)

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("❌ Cloud API hatası: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("❌ Cloud API %d: %s\n", resp.StatusCode, string(body))
		os.Exit(1)
	}

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok {
						fmt.Print(content)
						fullResponse.WriteString(content)
					}
				}
			}
		}
	}

	return fullResponse.String()
}

// ─── Veri Maskeleme ───────────────────────────────────────────────────────────

var maskPatterns = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	// Bearer token
	{regexp.MustCompile(`(?i)Bearer\s+[A-Za-z0-9\-_.]+`), "Bearer <REDACTED_TOKEN>"},
	// Authorization header value
	{regexp.MustCompile(`(?i)(authorization|auth)[=:\s]+[^\s]+`), "$1=<REDACTED_TOKEN>"},
	// password=... veya password: ...
	{regexp.MustCompile(`(?i)(password|passwd|pwd|secret|token|api[_-]?key)[=:\s]+[^\s]+`), "$1=<REDACTED>"},
	// IPv4 adresleri (private range öncelikli ama hepsini maskele)
	{regexp.MustCompile(`\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b`), "<REDACTED_IP>"},
	// Generic IPv4
	{regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`), "<REDACTED_IP>"},
	// AWS ARN
	{regexp.MustCompile(`arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d+:[^\s]+`), "<REDACTED_ARN>"},
}

func maskSensitiveData(input string) string {
	result := input
	for _, mp := range maskPatterns {
		result = mp.pattern.ReplaceAllString(result, mp.replacement)
	}
	return result
}

// ─── Araç & Dosya Tespiti ─────────────────────────────────────────────────────

func detectToolByKeywords(query string, tools map[string]ToolMeta) string {
	queryLower := strings.ToLower(query)
	queryWords := strings.FieldsFunc(queryLower, func(r rune) bool {
		return r == ' ' || r == '\'' || r == '"' || r == ',' || r == '.'
	})

	best, bestScore := "none", 0

	for _, meta := range tools {
		score := 0
		if strings.Contains(queryLower, strings.ToLower(meta.Name)) {
			score += 3
		}
		for _, kw := range meta.Keywords {
			kw = strings.TrimSpace(strings.ToLower(kw))
			if kw == "" {
				continue
			}
			for _, w := range queryWords {
				if w == kw {
					score += 2
					break
				}
			}
			if strings.Contains(queryLower, kw) {
				score += 1
			}
		}
		if score > bestScore {
			bestScore = score
			best = meta.Name
		}
	}

	if bestScore == 0 {
		return "none"
	}
	return best
}

func detectSubFileByKeywords(query string, meta ToolMeta) string {
	if len(meta.SubFiles) == 0 {
		return "general"
	}

	queryLower := strings.ToLower(query)
	queryWords := strings.FieldsFunc(queryLower, func(r rune) bool {
		return r == ' ' || r == '\'' || r == '"' || r == ',' || r == '.'
	})

	best, bestScore := "general", 0

	for fileName, keywordsRaw := range meta.SubFiles {
		score := 0
		for _, kw := range strings.Fields(keywordsRaw) {
			kw = strings.TrimSpace(strings.ToLower(kw))
			if kw == "" {
				continue
			}
			for _, w := range queryWords {
				if w == kw {
					score += 2
					break
				}
			}
			if strings.Contains(queryLower, kw) {
				score += 1
			}
		}
		if score > bestScore {
			bestScore = score
			best = fileName
		}
	}

	return best
}

func loadSubFileContext(meta ToolMeta, subFile string) string {
	path := filepath.Join(meta.Dir, subFile+".txt")
	content := tryReadFile(path)
	if content != "" {
		return content
	}
	return tryReadFile(filepath.Join(meta.Dir, "general.txt"))
}

// ─── Knowledge Yükleyici ─────────────────────────────────────────────────────

func listTools() map[string]ToolMeta {
	homeDir, _ := os.UserHomeDir()
	base := filepath.Join(homeDir, ".term-ai", "knowledge")

	entries, err := os.ReadDir(base)
	if err != nil {
		return map[string]ToolMeta{}
	}

	tools := make(map[string]ToolMeta)

	for _, e := range entries {
		if e.IsDir() {
			meta := loadToolMeta(filepath.Join(base, e.Name()), e.Name())
			tools[e.Name()] = meta
		} else if strings.HasSuffix(e.Name(), ".txt") {
			// Geriye dönük uyumluluk: düz .txt
			name := strings.TrimSuffix(e.Name(), ".txt")
			path := filepath.Join(base, e.Name())
			meta := ToolMeta{Name: name, Dir: base}
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
			tools[name] = meta
		}
	}
	return tools
}

func loadToolMeta(dir, name string) ToolMeta {
	meta := ToolMeta{Name: name, Dir: dir, SubFiles: make(map[string]string)}

	content := tryReadFile(filepath.Join(dir, "_meta.txt"))
	if content == "" {
		return meta
	}

	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# TANIM:") {
			meta.Description = strings.TrimSpace(strings.TrimPrefix(line, "# TANIM:"))
		} else if strings.HasPrefix(line, "# ANAHTAR KELİMELER:") {
			raw := strings.TrimSpace(strings.TrimPrefix(line, "# ANAHTAR KELİMELER:"))
			meta.Keywords = strings.Split(raw, ",")
		} else if strings.HasPrefix(line, "# DOSYALAR:") {
			raw := strings.TrimSpace(strings.TrimPrefix(line, "# DOSYALAR:"))
			for _, group := range strings.Split(raw, "|") {
				group = strings.TrimSpace(group)
				parts := strings.SplitN(group, "=", 2)
				if len(parts) == 2 {
					meta.SubFiles[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
				}
			}
		}
	}
	return meta
}

func getSmartContext(baseCmd, subCmd string) string {
	homeDir, _ := os.UserHomeDir()
	base := filepath.Join(homeDir, ".term-ai", "knowledge")

	toolDir := filepath.Join(base, baseCmd)
	if info, err := os.Stat(toolDir); err == nil && info.IsDir() {
		meta := loadToolMeta(toolDir, baseCmd)
		subFile := detectSubFileByKeywords(subCmd, meta)
		ctx := loadSubFileContext(meta, subFile)
		if ctx != "" {
			return ctx
		}
	}

	content := tryReadFile(filepath.Join(base, baseCmd+".txt"))
	if content == "" {
		return ""
	}
	return extractSection(content, subCmd)
}

func gatherAllGeneralSections(tools map[string]ToolMeta) string {
	var parts []string
	for name, meta := range tools {
		var content string
		if meta.SubFiles != nil {
			content = tryReadFile(filepath.Join(meta.Dir, "general.txt"))
		}
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
3. Bilinmesi gereken değerler (şifre, domain, IP) için <PLACEHOLDER> formatını kullan.

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

// ─── Dosya Okuyucular ─────────────────────────────────────────────────────────

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
	fmt.Printf("  ── Local ──────────────────────────\n")
	fmt.Printf("  model:          %s\n", cfg.Model)
	nlm := cfg.NLModel
	if nlm == "" { nlm = "(model kullanılıyor)" }
	fmt.Printf("  nl_model:       %s\n", nlm)
	fmt.Printf("  ollama_url:     %s\n", cfg.OllamaURL)
	fmt.Printf("\n  ── Cloud ──────────────────────────\n")
	provider := cfg.CloudProvider
	if provider == "" { provider = "(ayarlanmamış)" }
	fmt.Printf("  cloud_provider: %s\n", provider)
	cm := cfg.CloudModel
	if cm == "" { cm = "(varsayılan)" }
	fmt.Printf("  cloud_model:    %s\n", cm)
	key := cfg.CloudAPIKey
	if key != "" {
		key = key[:min(8, len(key))] + "..."
	} else {
		key = "(ayarlanmamış)"
	}
	fmt.Printf("  cloud_api_key:  %s\n", key)
	fmt.Printf("\n  language:       %s\n", cfg.Language)
	fmt.Printf("\nDüzenlemek için: %s\n", filepath.Join(termAIDir(), configFile))
}

func min(a, b int) int {
	if a < b { return a }
	return b
}

// ─── Geçmiş ───────────────────────────────────────────────────────────────────

func historyPath() string { return filepath.Join(termAIDir(), historyFile) }

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
		icon := "🔧"
		if e.Mode == "nl" {
			icon = "🧠"
		}
		if e.Mode == "chat" {
			icon = "💬"
		}
		backendIcon := "💻"
		if e.Backend == "cloud" {
			backendIcon = "☁️"
		}
		t, _ := time.Parse(time.RFC3339, e.Timestamp)
		fmt.Printf("%s %s %s [%s] %s\n   → %s\n\n",
			status, icon, backendIcon, t.Format("02 Jan 15:04"), e.Command, e.Suggestion)
	}
}
// ─── Training Data Collection ─────────────────────────────────────────────────

func trainingDataPath() string {
	return filepath.Join(termAIDir(), trainingDataFile)
}

func loadTrainingData() []TrainingExample {
	path := trainingDataPath()
	data, err := os.ReadFile(path)
	if err != nil {
		return []TrainingExample{}
	}
	var examples []TrainingExample
	json.Unmarshal(data, &examples)
	return examples
}

func saveTrainingExample(example TrainingExample) {
	examples := loadTrainingData()
	
	// Duplicate kontrolü (aynı instruction + output varsa ekleme)
	for _, ex := range examples {
		if ex.Instruction == example.Instruction && ex.Output == example.Output {
			return // Zaten var
		}
	}
	
	examples = append(examples, example)
	
	data, err := json.MarshalIndent(examples, "", "  ")
	if err != nil {
		return
	}
	
	os.WriteFile(trainingDataPath(), data, 0644)
}

func addApprovedCommand(query, command, tool string) {
	example := TrainingExample{
		Instruction: query,
		Output:      command,
		Tool:        tool,
		Timestamp:   time.Now().Format(time.RFC3339),
		Source:      "user_approved",
	}
	saveTrainingExample(example)
}

func showTrainingStats() {
	examples := loadTrainingData()
	
	if len(examples) == 0 {
		fmt.Println("📊 Training Data: Henüz onaylanmış komut yok")
		return
	}
	
	// Araç bazında istatistik
	toolStats := make(map[string]int)
	for _, ex := range examples {
		toolStats[ex.Tool]++
	}
	
	fmt.Printf("📊 Training Data İstatistikleri\n")
	fmt.Printf("   Toplam: %d onaylanmış komut\n\n", len(examples))
	fmt.Printf("   Araç Dağılımı:\n")
	for tool, count := range toolStats {
		fmt.Printf("   • %s: %d örnek\n", tool, count)
	}
	fmt.Printf("\n   Dosya: %s\n", trainingDataPath())
}


// ─── Index ─────────────────────────────────────────────────────────────────────────────

func runIndex(cfg Config, verbose bool) {
	if verbose {
		fmt.Printf("📦 Knowledge index’leniyor...\n")
		fmt.Printf("   Model: %s\n", defaultEmbedModel)
		fmt.Printf("   Ollama: %s\n\n", cfg.OllamaURL)
		fmt.Printf("   ℹ️  Yoksa kur: ollama pull %s\n\n", defaultEmbedModel)
	}

	if err := IndexKnowledge(cfg.OllamaURL, verbose); err != nil {
		fmt.Printf("\n❌ Index hatası: %v\n", err)
		os.Exit(1)
	}
}

// ─── Yardım ───────────────────────────────────────────────────────────────────

func showHelp() {
	fmt.Println(`🤖 term-ai — Terminal için AI Asistan

Kullanım:
  term-ai <komut> [argümanlar]     🔧 FIX: Komutu çalıştır, hata varsa düzelt
  term-ai "<doğal dil sorgusu>"    🧠 NL:  Komut bilmeden doğal dille sor
  term-ai --history                Son önerileri göster
  term-ai --config                 Mevcut ayarları göster
  term-ai --models                 Kurulu Ollama modellerini listele
  term-ai --index                  Knowledge dosyalarını vektörize et
  term-ai --training-stats         📊 Onaylanmış komut istatistikleri
  term-ai --no-history <komut>     Geçmişi atla (okuma + kaydetme yok)
  term-ai --cloud <sorgu>          Cloud API kullan (local yerine)
  term-ai --chat "<soru>"          Serbest sohbet modu (komut üretme yok)

FIX modu:
  term-ai git psuh origin main
  term-ai --cloud kubectl get pods -n producton    ← cloud ile

NL modu:
  term-ai "instana backendi production modda ayağa kaldır"
  term-ai --cloud "karmaşık bir bash scripti yaz"  ← cloud ile

Training Data:
  Onayladığınız her komut otomatik olarak ~/.term-ai/training_data.json'a kaydedilir.
  Bu dosya gelecekte model fine-tuning için kullanılabilir.

Config (~/.term-ai/config.json):
  local:  model, nl_model, ollama_url
  cloud:  cloud_provider (anthropic|gemini|openai), cloud_api_key, cloud_model`)
}

func showAvailableModels() {
	fmt.Print("📦 Kurulu Ollama modelleri:\n\n")
	cmd := exec.Command("ollama", "list")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Println("❌ 'ollama list' çalıştırılamadı.")
	}
	fmt.Printf("\nLocal fix modu (\"model\"):\n")
	fmt.Printf("  %-28s %s\n", "qwen2.5-coder:7b", "~5GB, önerilir")
	fmt.Printf("  %-28s %s\n", "qwen2.5-coder:14b", "~9GB, daha güçlü")
	fmt.Printf("\nLocal NL modu (\"nl_model\"):\n")
	fmt.Printf("  %-28s %s\n", "gemma3:1b", "~800MB ← önerilir")
	fmt.Printf("  %-28s %s\n", "gemma3:4b", "~3GB, daha kaliteli")
	fmt.Printf("\nCloud (\"cloud_provider\" + \"cloud_api_key\"):\n")
	fmt.Printf("  %-28s %s\n", "anthropic → claude-haiku-4-5", "hızlı, ucuz")
	fmt.Printf("  %-28s %s\n", "gemini → gemini-2.0-flash", "ücretsiz tier var")
	fmt.Printf("  %-28s %s\n", "openai → gpt-4o-mini", "dengeli")
}