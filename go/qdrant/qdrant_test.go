package qdrant_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/qdrant/qdrant-genkit/go/qdrant"
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

	g, err := genkit.Init(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	embedder := newFakeEmbedder()
	embedder.Register(d1, v1)
	embedder.Register(d2, v2)
	embedder.Register(d3, v3)

	cfg := qdrant.Config{
		GrpcHost:       "localhost",
		Port:           6334,
		CollectionName: collectionName,
		Embedder:       genkit.DefineEmbedder(g, "fake", "embedder3", embedder.Embed),
	}
	if err := qdrant.Init(ctx, g, cfg); err != nil {
		t.Fatal(err)
	}

	indexer := qdrant.Indexer(g, collectionName)
	err = indexer.Index(ctx, &ai.IndexerRequest{
		Documents: []*ai.Document{d1, d2, d3},
	})
	if err != nil {
		t.Fatalf("Index operation failed: %v", err)
	}

	retrieverOptions := &qdrant.RetrieverOptions{
		K: 2,
	}

	retriever := qdrant.Retriever(g, collectionName)
	retrieverResp, err := retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query:   d1,
		Options: retrieverOptions,
	})
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

func newFakeEmbedder() *embedder {
	return &embedder{
		registry: make(map[*ai.Document][]float32),
	}
}

func (e *embedder) Register(d *ai.Document, vals []float32) {
	e.registry[d] = vals
}

func (e *embedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	embeddings := make([]*ai.Embedding, len(req.Input))
	for i, doc := range req.Input {
		vals, ok := e.registry[doc]
		if !ok {
			return nil, errors.New("fake embedder called with unregistered document")
		}
		embeddings[i] = &ai.Embedding{
			Embedding: vals,
		}
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}
