import { EmbedderArgument } from '@genkit-ai/ai/embedder';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
  type RetrieverAction,
  type RetrieverReference,
} from '@genkit-ai/ai/retriever';
import type { QdrantClientParams, Schemas } from '@qdrant/js-client-rest';
import { QdrantClient } from '@qdrant/js-client-rest';
import { z, type Genkit } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { v5 as uuidv5 } from 'uuid';

const FilterType: z.ZodType<Schemas['Filter']> = z.any();
const PrefetchType: z.ZodType<Schemas['Prefetch'] | Schemas['Prefetch'][]> =
  z.any();
const QueryType: z.ZodType<Schemas['Query']> = z.any();

const QdrantRetrieverOptionsSchema: z.ZodObject<{
  k: z.ZodDefault<z.ZodNumber>;
  filter: z.ZodOptional<typeof FilterType>;
  scoreThreshold: z.ZodOptional<z.ZodNumber>;
  prefetch: z.ZodOptional<typeof PrefetchType>;
  query: z.ZodOptional<typeof QueryType>;
  groupBy: z.ZodOptional<z.ZodString>;
  groupSize: z.ZodOptional<z.ZodNumber>;
}> = CommonRetrieverOptionsSchema.extend({
  k: z.number().default(10),
  filter: FilterType.optional(),
  scoreThreshold: z.number().optional(),
  prefetch: PrefetchType.optional(),
  query: QueryType.optional(),
  // When set, results are grouped by this payload field (e.g. a document id or
  // category) via the Qdrant Grouping API, returning up to `groupSize` hits per
  // group. This diversifies results so a single over-represented group cannot
  // crowd out others. The group key is exposed on each returned document's
  // metadata as the reserved `_group` field (overrides any same-named field in
  // the document's own metadata). Documents are still returned as a flat list.
  groupBy: z.string().min(1).optional(),
  // Max hits per group when `groupBy` is set. Defaults to 3 (Qdrant default).
  groupSize: z.number().int().positive().optional(),
});

export const QdrantIndexerOptionsSchema = z.null().optional();

const CONTENT_PAYLOAD_KEY = 'content';
const METADATA_PAYLOAD_KEY = 'metadata';
const CONTENT_TYPE_KEY = '_content_type';
// Qdrant's own default for `group_size` when grouping; used to size the
// default prefetch candidate pool so reranking can fill `k` groups.
const DEFAULT_GROUP_SIZE = 3;

/**
 * Parameters for the Qdrant plugin.
 */
interface QdrantPluginParams<E extends z.ZodTypeAny = z.ZodTypeAny> {
  /**
   * Parameters for instantiating `QdrantClient`.
   */
  clientParams: QdrantClientParams;
  /**
   * Name of the Qdrant collection.
   */
  collectionName: string;
  /**
   * Embedder to use for the retriever and indexer.
   */
  embedder: EmbedderArgument<E>;
  /**
   * Addtional options for the embedder.
   */
  embedderOptions?: z.infer<E>;
  /**
   * Document content key in the Qdrant payload.
   * Default is 'content'.
   */
  contentPayloadKey?: string;
  /**
   * Document metadata key in the Qdrant payload.
   * Default is 'metadata'.
   */
  metadataPayloadKey?: string;
  /**
   * Document data type key in the Qdrant payload.
   * Default is '_content_type'.
   * This is used to store the type of content.
   */
  dataTypePayloadKey?: string;
  /**
   * Additional options when creating a collection.
   */
  collectionCreateOptions?: Schemas['CreateCollection'];
}

/**
 * qdrantRetrieverRef function creates a retriever for Qdrant.
 * @param params The params for the new Qdrant retriever
 * @param params.collectionName The collection name for the Qdrant retriever
 * @param params.displayName  A display name for the retriever. If not specified, the default label will be `Qdrant - <collectionName>`
 * @returns A reference to a Qdrant retriever.
 */
export const qdrantRetrieverRef = (
  collectionName: string,
  displayName: string | null = null,
): RetrieverReference<typeof QdrantRetrieverOptionsSchema> => {
  return retrieverRef({
    name: `qdrant/${collectionName}`,
    info: {
      label: displayName ?? `Qdrant - ${collectionName}`,
    },
    configSchema: QdrantRetrieverOptionsSchema,
  });
};

/**
 * qdrantIndexerRef function creates an indexer for Qdrant.
 * @param params The params for the new Qdrant indexer.
 * @param params.collectionName The collection name for the Qdrant indexer.
 * @param params.displayName  A display name for the indexer. If not specified, the default label will be `Qdrant - <collectionName>`
 * @returns A reference to a Qdrant indexer.
 */
export const qdrantIndexerRef = (
  collectionName: string,
  displayName: string | null = null,
) => {
  return indexerRef({
    name: `qdrant/${collectionName}`,
    info: {
      label: displayName ?? `Qdrant - ${collectionName}`,
    },
    configSchema: QdrantIndexerOptionsSchema,
  });
};

/**
 * Qdrant plugin that provides the Qdrant retriever
 * and indexer
 */
export function qdrant<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: QdrantPluginParams<EmbedderCustomOptions>[],
): GenkitPlugin {
  return genkitPlugin('qdrant', async (ai) => {
    params.forEach((p) => configureQdrantRetriever(ai, p));
    params.forEach((p) => configureQdrantIndexer(ai, p));
  });
}

export default qdrant;

export function configureQdrantRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: QdrantPluginParams<EmbedderCustomOptions>,
): RetrieverAction<typeof QdrantRetrieverOptionsSchema> {
  const {
    embedder,
    collectionName,
    embedderOptions,
    clientParams,
    contentPayloadKey,
    metadataPayloadKey,
  } = params;
  const client = new QdrantClient(clientParams);
  const contentKey = contentPayloadKey ?? CONTENT_PAYLOAD_KEY;
  const metadataKey = metadataPayloadKey ?? METADATA_PAYLOAD_KEY;
  const dataTypeKey = params.dataTypePayloadKey ?? CONTENT_TYPE_KEY;
  return ai.defineRetriever(
    {
      name: `qdrant/${collectionName}`,
      configSchema: QdrantRetrieverOptionsSchema,
    },
    async (content, options) => {
      await ensureCollection(params, false, ai);
      const queryEmbeddings = await ai.embed({
        embedder,
        content,
        options: embedderOptions,
      });
      const embedding = queryEmbeddings[0].embedding;
      const withPayload = [contentKey, metadataKey, dataTypeKey];
      // Shared mapper for both the flat and grouped paths: a scored point
      // (`{ payload, score }`) becomes a Genkit Document. `extraMetadata` lets
      // the grouped path attach the group key.
      const toDocument = (
        point: { payload?: Record<string, unknown> | null; score?: number },
        extraMetadata: Record<string, unknown> = {},
      ) => {
        const content = point.payload?.[contentKey] ?? '';
        const metadata = {
          ...((point.payload?.[metadataKey] as Record<string, unknown>) ?? {}),
          _similarityScore: point.score,
          ...extraMetadata,
        } as Record<string, unknown>;
        const dataType = point.payload?.[dataTypeKey] ?? 'text';
        return Document.fromData(
          content as string,
          dataType as string,
          metadata,
        ).toJSON();
      };

      // Prefetch/query reranking (formula boosting, fusion) applies to both the
      // flat and grouped paths: when `query` is set the embedded vector becomes
      // a prefetch (overridable) and `query` reranks the candidates.
      // For grouping, `k` counts groups (not points), so the default prefetch
      // must pull enough candidates to populate `k` groups of `groupSize` —
      // otherwise reranking sees too few points to fill the groups.
      const defaultPrefetchLimit = options.groupBy
        ? options.k * (options.groupSize ?? DEFAULT_GROUP_SIZE)
        : options.k;
      const prefetch = options.query
        ? (options.prefetch ?? {
            query: embedding,
            limit: defaultPrefetchLimit,
          })
        : undefined;
      const query = options.query ?? embedding;

      let documents: ReturnType<typeof toDocument>[];
      if (options.groupBy) {
        // Grouping API: up to `groupSize` hits per `groupBy` value, `k` groups.
        // Diversifies results across the facet so one over-represented group
        // (e.g. a large multi-chunk document, or a dominant category) cannot
        // crowd out the rest. The group key is exposed on each document's
        // metadata as `_group`; documents are still returned as a flat list.
        const groups = (
          await client.queryGroups(collectionName, {
            prefetch,
            query,
            group_by: options.groupBy,
            group_size: options.groupSize,
            limit: options.k,
            filter: options.filter,
            score_threshold: options.scoreThreshold,
            with_payload: withPayload,
            with_vector: false,
          })
        ).groups;
        documents = groups.flatMap((group) =>
          group.hits.map((hit) => toDocument(hit, { _group: group.id })),
        );
      } else {
        const results = (
          await client.query(collectionName, {
            prefetch,
            query,
            limit: options.k,
            filter: options.filter,
            score_threshold: options.scoreThreshold,
            with_payload: withPayload,
            with_vector: false,
          })
        ).points;
        documents = results.map((result) => toDocument(result));
      }
      return {
        documents,
      };
    },
  );
}

export function configureQdrantIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(ai: Genkit, params: QdrantPluginParams<EmbedderCustomOptions>) {
  const {
    embedder,
    collectionName,
    embedderOptions,
    clientParams,
    contentPayloadKey,
    metadataPayloadKey,
  } = params;
  const client = new QdrantClient(clientParams);
  const contentKey = contentPayloadKey ?? CONTENT_PAYLOAD_KEY;
  const metadataKey = metadataPayloadKey ?? METADATA_PAYLOAD_KEY;
  const dataTypeKey = params.dataTypePayloadKey ?? CONTENT_TYPE_KEY;
  return ai.defineIndexer(
    {
      name: `qdrant/${collectionName}`,
      configSchema: QdrantIndexerOptionsSchema,
    },
    async (docs, options) => {
      await ensureCollection(params, true, ai);
      const embeddings = await Promise.all(
        docs.map((doc) =>
          ai.embed({
            embedder,
            content: doc,
            options: embedderOptions,
          }),
        ),
      );
      const points = embeddings
        .map((embeddingArr, i) => {
          const doc = docs[i];
          const embeddingDocs = doc.getEmbeddingDocuments(embeddingArr);
          return embeddingArr.map((docEmbedding, j) => {
            const embeddingDoc = embeddingDocs[j] || {};
            const id = uuidv5(JSON.stringify(embeddingDoc), uuidv5.URL);
            return {
              id,
              vector: docEmbedding.embedding,
              payload: {
                [contentKey]: embeddingDoc.data,
                [metadataKey]: embeddingDoc.metadata,
                [dataTypeKey]: embeddingDoc.dataType,
              },
            };
          });
        })
        .reduce((acc, val) => acc.concat(val), []);
      await client.upsert(collectionName, { points });
    },
  );
}

/**
 * Helper function for creating a Qdrant collection.
 */
export async function createQdrantCollection<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: QdrantPluginParams<EmbedderCustomOptions>, ai) {
  const { embedder, embedderOptions, clientParams, collectionName } = params;
  const client = new QdrantClient(clientParams);
  let collectionCreateOptions = params.collectionCreateOptions;
  if (!collectionCreateOptions) {
    const embeddings = await ai.embed({
      embedder,
      content: 'SOME_TEXT',
      options: embedderOptions,
    });
    const vector = Array.isArray(embeddings)
      ? embeddings[0].embedding
      : embeddings.embedding;
    collectionCreateOptions = {
      vectors: {
        size: vector.length,
        distance: 'Cosine',
      },
    };
  }
  return await client.createCollection(collectionName, collectionCreateOptions);
}

/**
 * Helper function for deleting Qdrant collections.
 */
export async function deleteQdrantCollection(params: QdrantPluginParams) {
  const client = new QdrantClient(params.clientParams);
  return await client.deleteCollection(params.collectionName);
}

/**
 * Private helper for ensuring that a Qdrant collection exists.
 */
async function ensureCollection(
  params: QdrantPluginParams,
  createCollection = true,
  ai?,
) {
  const { clientParams, collectionName } = params;
  const client = new QdrantClient(clientParams);

  if ((await client.collectionExists(collectionName)).exists) {
    return;
  }

  if (createCollection) {
    await createQdrantCollection(params, ai);
  } else {
    throw new Error(
      `Collection ${collectionName} does not exist. Index some documents first.`,
    );
  }
}
