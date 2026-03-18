package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

const (
	ollamaModel = "qwen2.5-coder:7b"
	ollamaURL   = "http://localhost:11434/api/generate"
)

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type GenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Println("🤖 Kullanım: term-ai <hatalı_komut>")
		os.Exit(1)
	}

	fullCommand := strings.Join(args, " ")

	userCmd := exec.Command(args[0], args[1:]...)
	errorOutput, err := userCmd.CombinedOutput()
	if err == nil {
		fmt.Print(string(errorOutput))
		os.Exit(0)
	}

	errorMessage := string(errorOutput)

	var baseCmdArgs []string
	for _, arg := range args {
		if strings.HasPrefix(arg, "-") {
			break
		}
		baseCmdArgs = append(baseCmdArgs, arg)
	}

	if len(baseCmdArgs) == 0 {
		fmt.Println("🤖 Hata: Geçerli bir ana komut bulunamadı.")
		os.Exit(1)
	}

	// FIX 1: Komut sistemde yoksa (Typo veya kurulmamışsa) Help çekemeyiz
	baseCmdArgs = append(baseCmdArgs, "--help")
	helpCmd := exec.Command(baseCmdArgs[0], baseCmdArgs[1:]...)
	helpOutput, helpErr := helpCmd.CombinedOutput()
	
	if helpErr != nil && len(helpOutput) == 0 {
		fmt.Printf("\n⚠️ Hata: '%s' sisteminizde bulunamadı veya çalıştırılamadı. Kurulu olduğundan emin misiniz?\n", baseCmdArgs[0])
		os.Exit(1)
	}

	helpText := string(helpOutput)

	if len(helpText) > 4000 {
		helpText = helpText[:1500] + "\n\n...[METİN KISALTILDI]...\n\n" + helpText[len(helpText)-2500:]
	}

	prompt := fmt.Sprintf(`Sen uzman bir sistem mühendisisin.
Aşağıdaki HATA ÇIKTISI ve YARDIM METNİ'ni kullanarak doğru komutu bul.

Kurallar:
1. Sadece yardım metnindeki geçerli parametreleri kullan.
2. YAZIM HATASI KONTROLÜ: Hatalı komuttaki her kelimeyi yardım metnindeki geçerli kaynak/komut isimleriyle karşılaştır. 
   "nodef" → "nodes" gibi benzer kelime varsa DOĞRUDAN düzelt, başka komut önerme.
3. Eğer yazım hatası değil gerçekten eksik argüman ise, 'oc api-resources' gibi yardımcı komut öner.
4. Cevabını KESİNLİKLE aşağıdaki formatta ver:
AÇIKLAMA: <Kısa açıklama>
KOMUT: <Sadece çalıştırılabilir tam komut>

--- HATA ÇIKTISI ---
%s

--- YARDIM METNİ ---
%s

--- HATALI KOMUT ---
%s
`, strings.TrimSpace(errorMessage), strings.TrimSpace(helpText), fullCommand)

	fmt.Printf("🤖 Hata tespit edildi. Asistan analiz ediyor...\n\n")
	
	fullResponse := askOllama(prompt)

	cmdRegex := regexp.MustCompile("(?i)KOMUT:\\s*`?([^`\n]+)`?")
	matches := cmdRegex.FindStringSubmatch(fullResponse)

	// FIX 2: Model formatı bozarsa sessizce çıkma, kullanıcıya haber ver
	if len(matches) < 2 {
		fmt.Println("\n⚠️ Model uygun formatta cevap veremedi veya bir komut öneremedi.")
		os.Exit(1)
	}

	suggestedCmd := strings.TrimSpace(matches[1])
	
	dangerousPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)rm\s+-[a-z]*r[a-z]*f`),
		regexp.MustCompile(`(?i)>\s*/dev/[sh]`),
		regexp.MustCompile(`(?i)mkfs`),
		regexp.MustCompile(`\$\(.*\)`),
		regexp.MustCompile("`.*`"),
		regexp.MustCompile(`(?i)mv\s+/\s+`),
		regexp.MustCompile(`(?i)chmod\s+-R\s+777`),
	}

	for _, pattern := range dangerousPatterns {
		if pattern.MatchString(suggestedCmd) {
			fmt.Println("\n⛔ Güvenlik Duvarı: Tehlikeli pattern tespit edildi. Çalıştırılması engellendi.")
			os.Exit(1)
		}
	}

	fmt.Printf("\n▶ Bu komutu çalıştırayım mı? [e/h]: ")
	reader := bufio.NewReader(os.Stdin)
	answer, _ := reader.ReadString('\n')
	answer = strings.TrimSpace(strings.ToLower(answer))

	if answer == "e" || answer == "evet" {
		fmt.Println("▶▶ Çalıştırılıyor:", suggestedCmd)
		fmt.Println(strings.Repeat("-", 40))
		
		executeCmd := exec.Command("sh", "-c", suggestedCmd)
		executeCmd.Stdout = os.Stdout
		executeCmd.Stderr = os.Stderr
		executeCmd.Stdin = os.Stdin
		
		if err := executeCmd.Run(); err != nil {
			fmt.Printf("\n▶▶▶ Komut hata döndürdü: %v\n", err)
		}
	} else {
		fmt.Println("İptal edildi.")
	}
}

func askOllama(prompt string) string {
	reqBody := GenerateRequest{
		Model:  ollamaModel,
		Prompt: prompt,
		Stream: true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		fmt.Println("JSON hatası:", err)
		return ""
	}

	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("❌ Ollama'ya ulaşılamadı. 'ollama serve' çalışıyor mu?")
		return ""
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		var chunk GenerateResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		fmt.Print(chunk.Response)
		fullResponse.WriteString(chunk.Response)
		if chunk.Done {
			break
		}
	}
	fmt.Println() 
	return fullResponse.String()
}