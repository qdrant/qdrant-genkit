package qdrant

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/google/uuid"
	qclient "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

const provider = "qdrant"
const contentPayloadKey = "_content"
const metadataPayloadKey = "_metadata"

// New returns an [ai.DocumentStore] that uses Qdrant.
type Config struct {
	CollectionName  string
	GrpcHost        string
	Port            int
	ApiKey          string
	UseTls          bool
	ContentKey      string
	MetadataKey     string
	Embedder        *ai.EmbedderAction
	EmbedderOptions any
}

func Init(ctx context.Context, cfg Config) (err error) {

	defer func() {
		if err != nil {
			err = fmt.Errorf("qdrant.Init: %w", err)
		}
	}()

	var tlsCredential credentials.TransportCredentials

	if !cfg.UseTls && cfg.ApiKey != "" {
		log.Println("Warning: API key is set but TLS is not enabled. The API key will be sent in plaintext.")
		log.Println("May fail when using Qdrant cloud.")
	}

	if cfg.UseTls {
		tlsCredential = credentials.NewTLS(&tls.Config{})
	} else {
		tlsCredential = insecure.NewCredentials()
	}

	conn, err := grpc.NewClient(cfg.GrpcHost, grpc.WithTransportCredentials(tlsCredential), withApiKeyInterceptor(cfg.ApiKey))
	if err != nil {
		return fmt.Errorf("failed to connect to Qdrant: %v", err)
	}

	client, err := qclient.NewClient(&qclient.Config{
		Host:   cfg.GrpcHost,
		Port:   cfg.Port,
		APIKey: cfg.ApiKey,
		UseTLS: cfg.UseTls,
	})

	if err != nil {
		return fmt.Errorf("failed to instantiate Qdrant client: %w", err)
	}
	store := &docStore{
		client:     client,
		connection: conn,
	}

	name := cfg.CollectionName
	ai.DefineIndexer(provider, name, store.Index)
	ai.DefineRetriever(provider, name, store.Retrieve)
	return nil
}

// Indexer returns the indexer with the given collection name.
func Indexer(name string) *ai.IndexerAction {
	return ai.LookupIndexer(provider, name)
}

// Retriever returns the retriever with the given collection name.
func Retriever(name string) *ai.RetrieverAction {
	return ai.LookupRetriever(provider, name)
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
	connection         *grpc.ClientConn
	embedder           *ai.EmbedderAction
	embedderOptions    any
	contentPayloadKey  string
	metadataPayloadKey string
}

// Index implements the genkit Retriever.Index method.
func (ds *docStore) Index(ctx context.Context, req *ai.IndexerRequest) error {
	if len(req.Documents) == 0 {
		return nil
	}

	// Use the embedder to convert each Document into a vector.
	points := make([]*qclient.PointStruct, 0, len(req.Documents))
	for _, doc := range req.Documents {
		ereq := &ai.EmbedRequest{
			Document: doc,
			Options:  ds.embedderOptions,
		}
		vals, err := ai.Embed(ctx, ds.embedder, ereq)
		if err != nil {
			return fmt.Errorf("qdrant index embedding failed: %v", err)
		}

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
			Vectors: qclient.NewVectors(vals...),
			Payload: qclient.NewValueMap(map[string]any{
				contentPayloadKey:  sb.String(),
				metadataPayloadKey: doc.Metadata,
			}),
		}
		points = append(points, point)
	}

	_, err := ds.client.Upsert(ctx, &qclient.UpsertPoints{
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
		Document: req.Document,
		Options:  ds.embedderOptions,
	}
	vector, err := ai.Embed(ctx, ds.embedder, ereq)
	if err != nil {
		return nil, fmt.Errorf("qdrant retrieve embedding failed: %v", err)
	}

	response, err := ds.client.Query(context.TODO(), &qclient.QueryPoints{
		CollectionName: ds.collectionName,
		Query:          qclient.NewQuery(vector...),
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

// Appends "api-key" to the metadata for authentication
func withApiKeyInterceptor(apiKey string) grpc.DialOption {
	return grpc.WithUnaryInterceptor(func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		newCtx := metadata.AppendToOutgoingContext(ctx, "api-key", apiKey)
		return invoker(newCtx, method, req, reply, cc, opts...)
	})
}
