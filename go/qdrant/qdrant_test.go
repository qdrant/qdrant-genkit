package qdrant_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/qdrant/genkitx-qdrant/go/qdrant"
)

func TestGenkit(t *testing.T) {

	ctx := context.Background()

	collectionName := "test-genkitx-qdrant"

	dim := 1536

	v1 := make([]float32, dim)
	v2 := make([]float32, dim)
	v3 := make([]float32, dim)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i)
		v3[i] = float32(dim - i)
	}
	v2[0] = 1

	d1 := ai.DocumentFromText("hello1", nil)
	d2 := ai.DocumentFromText("hello2", nil)
	d3 := ai.DocumentFromText("goodbye", nil)

	embedder := newFakeEmbedder()
	embedder.Register(d1, v1)
	embedder.Register(d2, v2)
	embedder.Register(d3, v3)

	cfg := qdrant.Config{
		GrpcHost:       "localhost",
		Embedder:       ai.DefineEmbedder("fake", "embedder3", embedder.Embed),
		CollectionName: collectionName,
	}
	if err := qdrant.Init(ctx, cfg); err != nil {
		t.Fatal(err)
	}

	err := ai.Index(ctx, qdrant.Indexer(collectionName), ai.WithIndexerDocs(d1, d2, d3))
	if err != nil {
		t.Fatalf("Index operation failed: %v", err)
	}

	retrieverOptions := &qdrant.RetrieverOptions{
		K: 2,
	}

	retrieverResp, err := ai.Retrieve(ctx, qdrant.Retriever(collectionName), ai.WithRetrieverDoc(d1), ai.WithRetrieverOpts(retrieverOptions))
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

type embedder struct {
	registry map[*ai.Document][]float32
}

// New returns a new fake embedder.
func newFakeEmbedder() *embedder {
	return &embedder{
		registry: make(map[*ai.Document][]float32),
	}
}

// Register records the value to return for a Document.
func (e *embedder) Register(d *ai.Document, vals []float32) {
	e.registry[d] = vals
}

func (e *embedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	embeddings := make([]*ai.DocumentEmbedding, len(req.Documents))
	for _, doc := range req.Documents {
		vals, ok := e.registry[doc]
		if !ok {
			return nil, errors.New("fake embedder called with unregistered document")
		}
		embeddings = append(embeddings, &ai.DocumentEmbedding{
			Embedding: vals,
		})
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}
