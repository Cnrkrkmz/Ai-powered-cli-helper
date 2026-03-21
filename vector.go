package main

// vector.go — Semantic search için embedding ve cosine similarity
//
// Mimari:
//   1. IndexKnowledge(): knowledge/ altındaki tüm .txt dosyalarını okur,
//      Ollama /api/embeddings endpoint'i ile vektörize eder,
//      ~/.term-ai/embeddings.json'a kaydeder.
//      Sadece değişen dosyaları yeniden index'ler (mtime kontrolü).
//
//   2. SearchSimilar(): kullanıcı sorgusunu vektörize eder,
//      embeddings.json'daki tüm vektörlerle cosine similarity hesaplar,
//      en yüksek skorlu (araç, dosya) çiftini döner.
//
// Gerekli model: ollama pull nomic-embed-text

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ─── Veri Yapıları ────────────────────────────────────────────────────────────

// EmbeddingEntry: bir dosyanın vektörünü ve metadata'sını tutar
type EmbeddingEntry struct {
	Tool      string    `json:"tool"`       // örn: "stanctl"
	SubFile   string    `json:"sub_file"`   // örn: "airgapped"
	FilePath  string    `json:"file_path"`  // tam yol
	ModTime   time.Time `json:"mod_time"`   // son değişiklik zamanı
	Vector    []float64 `json:"vector"`     // embedding vektörü
	// İlk 200 karakter — debug için
	Preview   string    `json:"preview"`
}

// EmbeddingIndex: embeddings.json dosyasının tamamı
type EmbeddingIndex struct {
	EmbedModel string           `json:"embed_model"`
	UpdatedAt  time.Time        `json:"updated_at"`
	Entries    []EmbeddingEntry `json:"entries"`
}

// SearchResult: SearchSimilar'ın döndürdüğü sonuç
type SearchResult struct {
	Tool     string
	SubFile  string
	FilePath string
	Score    float64
}

// Ollama embedding request/response
type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

// ─── Sabitler ────────────────────────────────────────────────────────────────

const (
	defaultEmbedModel  = "nomic-embed-text"
	embeddingsFile     = "embeddings.json"
	// Cosine similarity eşiği: bu değerin altındaki sonuçları görmezden gel
	similarityThreshold = 0.3
)

// ─── Ana Fonksiyonlar ─────────────────────────────────────────────────────────

// IndexKnowledge: knowledge/ dizinini tarar, değişen dosyaları vektörize eder.
// verbose=true ise ilerlemeyi ekrana yazar.
func IndexKnowledge(ollamaURL string, verbose bool) error {
	homeDir, _ := os.UserHomeDir()
	knowledgeDir := filepath.Join(homeDir, ".term-ai", "knowledge")
	indexPath := filepath.Join(homeDir, ".term-ai", embeddingsFile)

	// Mevcut index'i yükle
	index := loadIndex(indexPath)
	existingMap := buildExistingMap(index)

	// knowledge/ altındaki tüm .txt dosyalarını topla
	allFiles, err := collectTextFiles(knowledgeDir)
	if err != nil {
		return fmt.Errorf("knowledge dizini okunamadı: %w", err)
	}

	if len(allFiles) == 0 {
		return fmt.Errorf("~/.term-ai/knowledge/ dizininde hiç .txt dosyası bulunamadı")
	}

	newEntries := []EmbeddingEntry{}
	indexedCount := 0
	skippedCount := 0

	for _, fp := range allFiles {
		info, err := os.Stat(fp.path)
		if err != nil {
			continue
		}
		modTime := info.ModTime()

		// Daha önce index'lenmiş ve değişmemiş mi?
		key := fp.path
		if existing, ok := existingMap[key]; ok {
			if !modTime.After(existing.ModTime) {
				// Değişmemiş, eski entry'yi koru
				newEntries = append(newEntries, existing)
				skippedCount++
				continue
			}
		}

		// Dosyayı oku
		content, err := os.ReadFile(fp.path)
		if err != nil {
			continue
		}

		// _meta.txt dosyalarını atla (bunlar araç tespiti için, içerik değil)
		if filepath.Base(fp.path) == "_meta.txt" {
			continue
		}

		text := string(content)
		if strings.TrimSpace(text) == "" {
			continue
		}

		if verbose {
			fmt.Printf("  📄 Index'leniyor: %s/%s...", fp.tool, fp.subFile)
		}

		// Vektörize et
		vector, err := getEmbedding(ollamaURL, defaultEmbedModel, text)
		if err != nil {
			if verbose {
				fmt.Printf(" ❌ hata: %v\n", err)
			}
			continue
		}

		preview := text
		if len(preview) > 200 {
			preview = preview[:200]
		}

		entry := EmbeddingEntry{
			Tool:     fp.tool,
			SubFile:  fp.subFile,
			FilePath: fp.path,
			ModTime:  modTime,
			Vector:   vector,
			Preview:  preview,
		}
		newEntries = append(newEntries, entry)
		indexedCount++

		if verbose {
			fmt.Printf(" ✅\n")
		}
	}

	// Güncel index'i kaydet
	newIndex := EmbeddingIndex{
		EmbedModel: defaultEmbedModel,
		UpdatedAt:  time.Now(),
		Entries:    newEntries,
	}

	if err := saveIndex(indexPath, newIndex); err != nil {
		return fmt.Errorf("index kaydedilemedi: %w", err)
	}

	if verbose {
		fmt.Printf("\n✅ Index tamamlandı: %d yeni, %d değişmemiş (%d toplam)\n",
			indexedCount, skippedCount, len(newEntries))
	}

	return nil
}

// SearchSimilar: sorguyu vektörize eder, en benzer dosyayı döner.
// index boşsa veya eşik altındaysa (tool="none", subFile="") döner.
func SearchSimilar(ollamaURL, query string) SearchResult {
	homeDir, _ := os.UserHomeDir()
	indexPath := filepath.Join(homeDir, ".term-ai", embeddingsFile)

	index := loadIndex(indexPath)
	if len(index.Entries) == 0 {
		return SearchResult{Tool: "none"}
	}

	// Sorguyu vektörize et
	queryVec, err := getEmbedding(ollamaURL, defaultEmbedModel, query)
	if err != nil {
		// Embedding başarısız → keyword'e düş (caller halleder)
		return SearchResult{Tool: "none"}
	}

	// En yüksek cosine similarity'yi bul
	bestScore := -1.0
	bestEntry := EmbeddingEntry{}

	for _, entry := range index.Entries {
		score := cosineSimilarity(queryVec, entry.Vector)
		if score > bestScore {
			bestScore = score
			bestEntry = entry
		}
	}

	if bestScore < similarityThreshold {
		return SearchResult{Tool: "none"}
	}

	return SearchResult{
		Tool:     bestEntry.Tool,
		SubFile:  bestEntry.SubFile,
		FilePath: bestEntry.FilePath,
		Score:    bestScore,
	}
}

// IsIndexStale: index'in güncel olup olmadığını kontrol eder.
// Herhangi bir .txt dosyası index'ten daha yeniyse true döner.
func IsIndexStale(ollamaURL string) bool {
	homeDir, _ := os.UserHomeDir()
	knowledgeDir := filepath.Join(homeDir, ".term-ai", "knowledge")
	indexPath := filepath.Join(homeDir, ".term-ai", embeddingsFile)

	indexInfo, err := os.Stat(indexPath)
	if err != nil {
		return true // index yok → stale
	}
	indexTime := indexInfo.ModTime()

	allFiles, err := collectTextFiles(knowledgeDir)
	if err != nil {
		return false
	}

	for _, fp := range allFiles {
		if filepath.Base(fp.path) == "_meta.txt" {
			continue
		}
		info, err := os.Stat(fp.path)
		if err != nil {
			continue
		}
		if info.ModTime().After(indexTime) {
			return true
		}
	}
	return false
}

// ─── Embedding ───────────────────────────────────────────────────────────────

func getEmbedding(ollamaURL, model, text string) ([]float64, error) {
	// Uzun metinleri kırp (nomic-embed-text max ~8192 token)
	if len(text) > 6000 {
		text = text[:6000]
	}

	reqBody := EmbedRequest{
		Model:  model,
		Prompt: text,
	}

	jsonData, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 30 * time.Second}

	// Ollama embedding endpoint: /api/embeddings (base URL'den türet)
	embedURL := strings.TrimSuffix(ollamaURL, "/api/generate") + "/api/embeddings"

	resp, err := client.Post(embedURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("ollama embedding isteği başarısız: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("ollama embedding %d döndü", resp.StatusCode)
	}

	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, err
	}

	if len(embedResp.Embedding) == 0 {
		return nil, fmt.Errorf("boş embedding döndü")
	}

	return embedResp.Embedding, nil
}

// ─── Cosine Similarity ────────────────────────────────────────────────────────

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ─── Dosya Tarayıcı ───────────────────────────────────────────────────────────

type fileRef struct {
	path    string
	tool    string
	subFile string
}

// collectTextFiles: knowledge/ altındaki tüm .txt dosyalarını toplar.
// Hem klasör yapısını (stanctl/cluster.txt) hem düz .txt'yi destekler.
func collectTextFiles(knowledgeDir string) ([]fileRef, error) {
	entries, err := os.ReadDir(knowledgeDir)
	if err != nil {
		return nil, err
	}

	var files []fileRef

	for _, e := range entries {
		if e.IsDir() {
			// Klasör bazlı: stanctl/cluster.txt → tool=stanctl, subFile=cluster
			toolDir := filepath.Join(knowledgeDir, e.Name())
			subEntries, err := os.ReadDir(toolDir)
			if err != nil {
				continue
			}
			for _, sub := range subEntries {
				if sub.IsDir() || !strings.HasSuffix(sub.Name(), ".txt") {
					continue
				}
				subFile := strings.TrimSuffix(sub.Name(), ".txt")
				files = append(files, fileRef{
					path:    filepath.Join(toolDir, sub.Name()),
					tool:    e.Name(),
					subFile: subFile,
				})
			}
		} else if strings.HasSuffix(e.Name(), ".txt") {
			// Düz .txt: git.txt → tool=git, subFile=general
			name := strings.TrimSuffix(e.Name(), ".txt")
			files = append(files, fileRef{
				path:    filepath.Join(knowledgeDir, e.Name()),
				tool:    name,
				subFile: "general",
			})
		}
	}

	return files, nil
}

// ─── Index Okuma/Yazma ────────────────────────────────────────────────────────

func loadIndex(path string) EmbeddingIndex {
	data, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingIndex{}
	}
	var index EmbeddingIndex
	json.Unmarshal(data, &index)
	return index
}

func saveIndex(path string, index EmbeddingIndex) error {
	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func buildExistingMap(index EmbeddingIndex) map[string]EmbeddingEntry {
	m := make(map[string]EmbeddingEntry)
	for _, e := range index.Entries {
		m[e.FilePath] = e
	}
	return m
}