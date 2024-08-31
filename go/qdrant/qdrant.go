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
	pb "github.com/qdrant/go-client/qdrant"
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
	ApiKey          string
	UseTls          bool
	ContentKey      string
	MetadataKey     string
	Embedder        ai.Embedder
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

	store := &docStore{
		collections_client: pb.NewCollectionsClient(conn),
		points_client:      pb.NewPointsClient(conn),
		service_client:     pb.NewQdrantClient(conn),
		connection:         conn,
	}

	name := cfg.CollectionName
	ai.DefineIndexer(provider, name, store.Index)
	ai.DefineRetriever(provider, name, store.Retrieve)
	return nil
}

// Indexer returns the indexer with the given collection name.
func Indexer(name string) ai.Indexer {
	return ai.LookupIndexer(provider, name)
}

// Retriever returns the retriever with the given collection name.
func Retriever(name string) ai.Retriever {
	return ai.LookupRetriever(provider, name)
}

type IndexerOptions struct{}

type RetrieverOptions struct {
	Filter pb.Filter
	K      int // maximum number of values to retrieve
}

// docStore implements the genkit [ai.DocumentStore] interface.
type docStore struct {
	collectionName     string
	collections_client pb.CollectionsClient
	points_client      pb.PointsClient
	service_client     pb.QdrantClient
	connection         *grpc.ClientConn
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

	// Use the embedder to convert each Document into a vector.
	points := make([]*pb.PointStruct, 0, len(req.Documents))
	ereq := &ai.EmbedRequest{
		Documents: req.Documents,
		Options:   ds.embedderOptions,
	}
	eres, err := ds.embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("qdrant index embedding failed: %v", err)
	}
	for i, res := range eres.Embeddings {
		id, err := generatePointId(req.Documents[i])
		if err != nil {
			return err
		}

		metadata := newValueMap(req.Documents[i].Metadata)

		var sb strings.Builder
		for _, p := range req.Documents[i].Content {
			sb.WriteString(p.Text)
		}

		point := &pb.PointStruct{
			Id: &pb.PointId{
				PointIdOptions: &pb.PointId_Uuid{
					Uuid: id,
				},
			},
			Vectors: &pb.Vectors{
				VectorsOptions: &pb.Vectors_Vector{
					Vector: &pb.Vector{
						Data: res.Embedding,
					},
				},
			},
			Payload: map[string]*pb.Value{
				contentPayloadKey: {
					Kind: &pb.Value_StringValue{
						StringValue: sb.String(),
					},
				},
				metadataPayloadKey: {
					Kind: &pb.Value_StructValue{
						StructValue: &pb.Struct{
							Fields: metadata,
						},
					},
				},
			}}
		points = append(points, point)
	}

	_, err = ds.points_client.Upsert(ctx, &pb.UpsertPoints{
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
		filter *pb.Filter
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
		Documents: []*ai.Document{req.Document},
		Options:   ds.embedderOptions,
	}
	vector, err := ds.embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("qdrant retrieve embedding failed: %v", err)
	}

	response, err := ds.points_client.Search(context.TODO(), &pb.SearchPoints{
		CollectionName: ds.collectionName,
		Vector:         vector.Embeddings[0].Embedding,
		Limit:          uint64(limit),
		Filter:         filter,
		WithPayload: &pb.WithPayloadSelector{
			SelectorOptions: &pb.WithPayloadSelector_Include{
				Include: &pb.PayloadIncludeSelector{
					Fields: []string{ds.contentPayloadKey, ds.metadataPayloadKey},
				},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	var docs []*ai.Document
	for _, result := range response.Result {
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
