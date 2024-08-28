package qdrant_test

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/qdrant/genkitx-qdrant/go/qdrant"
)

func TestGenkit(t *testing.T) {

	ctx := context.Background()

	collectionName := "test-genkitx-qdrant"

	dim := 1536

	d1 := ai.DocumentFromText("hello1", nil)

	cfg := qdrant.Config{
		GrpcHost: "localhost",
		Embedder: ai.DefineEmbedder("fake", "embedder3", func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			embeddings := make([]*ai.DocumentEmbedding, len(req.Documents))
			for i := range req.Documents {
				embeddings[i] = &ai.DocumentEmbedding{
					Embedding: make([]float32, dim),
				}
			}
			return &ai.EmbedResponse{
				Embeddings: embeddings,
			}, nil
		}),
	}
	if err := qdrant.Init(ctx, cfg); err != nil {
		t.Fatal(err)
	}

	indexerOptions := &qdrant.IndexerOptions{}

	err := ai.Index(ctx, qdrant.Indexer(collectionName), ai.WithIndexerDocs(d1), ai.WithIndexerOpts(indexerOptions))
	if err != nil {
		t.Fatalf("Index operation failed: %v", err)
	}

	retrieverOptions := &qdrant.RetrieverOptions{
		K: 2,
	}

	retrieverResp, err := ai.Retrieve(ctx, qdrant.Retriever(collectionName), ai.WithRetrieverText("hello"), ai.WithRetrieverOpts(retrieverOptions))
	if err != nil {
		t.Fatalf("Retrieve operation failed: %v", err)
	}

	docs := retrieverResp.Documents
	if len(docs) != 2 {
		t.Errorf("got %d results, expected 2", len(docs))
	}
	for _, d := range docs {
		text := d.Content[0].Text
		if !strings.HasPrefix(text, "hello") {
			t.Errorf("returned doc text %q does not start with %q", text, "hello")
		}
	}
}
