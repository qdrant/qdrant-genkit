package qdrant

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/uuid"
	qclient "github.com/qdrant/go-client/qdrant"
)

const provider = "qdrant"
const defaultContentKey = "_content"
const defaultMetadataKey = "_metadata"

// Config provides configuration options for Qdrant.
type Config struct {
	CollectionName  string
	GrpcHost        string
	Port            int
	ApiKey          string
	UseTls          bool
	ContentKey      string // Optional: defaults to "_content"
	MetadataKey     string // Optional: defaults to "_metadata"
	Embedder        ai.Embedder
	EmbedderOptions any
}

// Init initializes the Qdrant plugin.
func Init(ctx context.Context, g *genkit.Genkit, cfg Config) (err error) {
	client, err := qclient.NewClient(&qclient.Config{
		Host:   cfg.GrpcHost,
		Port:   cfg.Port,
		APIKey: cfg.ApiKey,
		UseTLS: cfg.UseTls,
	})

	if err != nil {
		return fmt.Errorf("failed to instantiate Qdrant client: %w", err)
	}

	contentKey := cfg.ContentKey
	if contentKey == "" {
		contentKey = defaultContentKey
	}
	metadataKey := cfg.MetadataKey
	if metadataKey == "" {
		metadataKey = defaultMetadataKey
	}

	store := &docStore{
		client:             client,
		collectionName:     cfg.CollectionName,
		embedder:           cfg.Embedder,
		embedderOptions:    cfg.EmbedderOptions,
		contentPayloadKey:  contentKey,
		metadataPayloadKey: metadataKey,
	}

	name := cfg.CollectionName
	genkit.DefineIndexer(g, provider, name, store.Index)
	genkit.DefineRetriever(g, provider, name, store.Retrieve)
	return nil
}

// Indexer returns the indexer with the given collection name.
func Indexer(g *genkit.Genkit, name string) ai.Indexer {
	return genkit.LookupIndexer(g, provider, name)
}

// Retriever returns the retriever with the given collection name.
func Retriever(g *genkit.Genkit, name string) ai.Retriever {
	return genkit.LookupRetriever(g, provider, name)
}

type IndexerOptions struct{}

type RetrieverOptions struct {
	Filter qclient.Filter
	K      int // maximum number of values to retrieve
}

// docStore implements the genkit [ai.DocumentStore] interface.
type docStore struct {
	collectionName     string
	client             *qclient.Client
	embedder           ai.Embedder
	embedderOptions    any
	contentPayloadKey  string
	metadataPayloadKey string
}

// Index implements the genkit Retriever.Index method.
func (ds *docStore) Index(ctx context.Context, req *ai.IndexerRequest) error {
	if len(req.Documents) == 0 {
		return nil
	}

	ereq := &ai.EmbedRequest{
		Input:   req.Documents,
		Options: ds.embedderOptions,
	}
	vals, err := ds.embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("qdrant index embedding failed: %v", err)
	}

	// Use the embedder to convert each Document into a vector.
	points := make([]*qclient.PointStruct, 0, len(req.Documents))
	for i, doc := range req.Documents {
		id, err := generatePointId(doc)
		if err != nil {
			return err
		}

		var sb strings.Builder
		for _, p := range doc.Content {
			sb.WriteString(p.Text)
		}

		point := &qclient.PointStruct{
			Id:      qclient.NewID(id),
			Vectors: qclient.NewVectors(vals.Embeddings[i].Embedding...),
			Payload: qclient.NewValueMap(map[string]any{
				ds.contentPayloadKey:  sb.String(),
				ds.metadataPayloadKey: doc.Metadata,
			}),
		}
		points = append(points, point)
	}

	_, err = ds.client.Upsert(ctx, &qclient.UpsertPoints{
		CollectionName: ds.collectionName,
		Points:         points,
	})

	if err != nil {
		return fmt.Errorf("qdrant index upsert failed: %v", err)
	}

	return nil
}

// Retrieve implements the genkit Retriever.Retrieve method.
func (ds *docStore) Retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	var (
		filter *qclient.Filter
		limit  int
	)
	if req.Options != nil {
		ropt, ok := req.Options.(*RetrieverOptions)
		if !ok {
			return nil, fmt.Errorf("qdrant.Retrieve options have type %T, want %T", req.Options, &RetrieverOptions{})
		}
		filter = &ropt.Filter
		limit = ropt.K
	}

	// Use the embedder to convert the document we want to
	// retrieve into a vector.
	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.embedderOptions,
	}
	vectors, err := ds.embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("qdrant retrieve embedding failed: %v", err)
	}

	response, err := ds.client.Query(context.TODO(), &qclient.QueryPoints{
		CollectionName: ds.collectionName,
		Query:          qclient.NewQuery(vectors.Embeddings[0].Embedding...),
		Limit:          qclient.PtrOf(uint64(limit)),
		Filter:         filter,
		WithPayload:    qclient.NewWithPayloadInclude(ds.contentPayloadKey, ds.metadataPayloadKey),
	})
	if err != nil {
		return nil, err
	}

	var docs []*ai.Document
	for _, result := range response {
		content := result.Payload[ds.contentPayloadKey].GetStringValue()
		if content == "" {
			return nil, errors.New("qdrant retrieve failed to fetch original document text")
		}

		metadata := make(map[string]any)
		for k, v := range result.Payload[ds.metadataPayloadKey].GetStructValue().Fields {
			metadata[k] = v
		}

		d := ai.DocumentFromText(content, metadata)
		docs = append(docs, d)
	}

	ret := &ai.RetrieverResponse{
		Documents: docs,
	}
	return ret, nil
}

// Generates a deterministic UUID and returns the string representation.
// Qdrant only allows UUIDs and positive integers as point IDs.
func generatePointId(doc *ai.Document) (string, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return "", fmt.Errorf("qdrant: error marshaling document: %v", err)
	}
	uuid := uuid.NewSHA1(uuid.NameSpaceDNS, b)
	return uuid.String(), nil
}
