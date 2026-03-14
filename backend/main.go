package main

import (
	"context"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/vishwas/backend/internal/gen/v1"
)

type Orchestrator struct {
	intelligenceClient pb.IntelligenceServiceClient
}

func main() {
	if err := godotenv.Load("../.env"); err != nil {
		log.Println("Note: No root .env file found")
	}

	intelligenceAddr := os.Getenv("INTELLIGENCE_SERVICE_ADDR")
	if intelligenceAddr == "" {
		intelligenceAddr = "localhost:50051"
	}

	log.Printf("Bridge: Connecting to Brain at %s...", intelligenceAddr)
	conn, err := grpc.Dial(intelligenceAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Bridge: Could not connect to Intelligence service: %v", err)
	}
	defer conn.Close()

	orch := &Orchestrator{
		intelligenceClient: pb.NewIntelligenceServiceClient(conn),
	}

	r := gin.Default()
	r.Use(cors.Default())

	api := r.Group("/api")
	{
		api.POST("/analyze/text", orch.handleAnalyzeText)
		api.POST("/analyze/image", orch.handleAnalyzeFile)
		api.POST("/analyze/document", orch.handleAnalyzeFile)
		api.POST("/report", orch.handleReport)
	}

	port := os.Getenv("BACKEND_PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Bridge: Go Orchestrator listening on :%s...", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal(err)
	}
}

func (o *Orchestrator) handleAnalyzeText(c *gin.Context) {
	text := c.PostForm("text")
	sourceURL := c.PostForm("source_url")

	log.Printf("Bridge: [TEXT] Analysis request (%d chars)", len(text))

	if text == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Empty text"})
		return
	}

	// Increased timeout to 2 minutes for RAG
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	resp, err := o.intelligenceClient.AnalyzeContent(ctx, &pb.AnalyzeRequest{
		Text:      text,
		SourceUrl: sourceURL,
	})

	if err != nil {
		log.Printf("Bridge: [TEXT] Brain error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Brain failed: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, resp)
}

func (o *Orchestrator) handleAnalyzeFile(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		log.Printf("Bridge: [FILE] Request error: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	
	textContext := c.PostForm("text")
	sourceURL := c.PostForm("source_url")

	log.Printf("Bridge: [FILE] Analysis request: %s (%d bytes)", file.Filename, file.Size)

	f, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open file"})
		return
	}
	defer f.Close()

	buf, err := io.ReadAll(f)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file"})
		return
	}

	// Increased timeout to 2 minutes for RAG
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	resp, err := o.intelligenceClient.AnalyzeContent(ctx, &pb.AnalyzeRequest{
		Text:       textContext,
		ImageBytes: buf,
		SourceUrl:  sourceURL,
		Filename:   file.Filename,
	})

	if err != nil {
		log.Printf("Bridge: [FILE] Brain error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Brain failed: " + err.Error()})
		return
	}

	log.Printf("Bridge: [FILE] Success. Score: %.1f", resp.Score)
	c.JSON(http.StatusOK, resp)
}

func (o *Orchestrator) handleReport(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "Report received by bridge"})
}
