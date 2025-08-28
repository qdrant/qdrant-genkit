import { EmbedderArgument } from '@genkit-ai/ai/embedder';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from '@genkit-ai/ai/retriever';
import type { QdrantClientParams, Schemas } from '@qdrant/js-client-rest';
import { QdrantClient } from '@qdrant/js-client-rest';
import { z, type Genkit } from 'genkit';
import { genkitPlugin } from 'genkit/plugin';
import { v5 as uuidv5 } from 'uuid';

const FilterType: z.ZodType<Schemas['Filter']> = z.any();

const QdrantRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().default(10),
  filter: FilterType.optional(),
  scoreThreshold: z.number().optional(),
});

export const QdrantIndexerOptionsSchema = z.null().optional();

const CONTENT_PAYLOAD_KEY = 'content';
const METADATA_PAYLOAD_KEY = 'metadata';
const CONTENT_TYPE_KEY = '_content_type';

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
export const qdrantRetrieverRef = (collectionName: string, displayName: string | null = null) => {
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
export const qdrantIndexerRef = (collectionName: string, displayName: string | null = null) => {
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
) {
  return genkitPlugin('qdrant', async (ai) => {
    params.forEach((p) => configureQdrantRetriever(ai, p));
    params.forEach((p) => configureQdrantIndexer(ai, p));
  });
}

export default qdrant;

export function configureQdrantRetriever<
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
      const results = (
        await client.query(collectionName, {
          query: queryEmbeddings[0].embedding,
          limit: options.k,
          filter: options.filter,
          score_threshold: options.scoreThreshold,
          with_payload: [contentKey, metadataKey, dataTypeKey],
          with_vector: false,
        })
      ).points;
      const documents = results.map((result) => {
        const content = result.payload?.[contentKey] ?? '';
        const metadata = {
          ...(result.payload?.[metadataKey] ?? {}),
          _similarityScore: result.score,
        } as Record<string, unknown>;
        const dataType = result.payload?.[dataTypeKey] ?? 'text';
        return Document.fromData(
          content as string,
          dataType as string,
          metadata as Record<string, unknown>,
        ).toJSON();
      });
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
