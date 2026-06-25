# Qdrant plugin

The Qdrant plugin provides [Genkit](https://firebase.google.com/docs/genkit) indexer and retriever implementations in JS and Go that use the [Qdrant](https://qdrant.tech/).

## Installation

```bash
npm i genkitx-qdrant
```

## Configuration

To use this plugin, specify it when you call `configureGenkit()`:

```js
import { qdrant } from 'genkitx-qdrant';

const ai = genkit({
  plugins: [
    qdrant([
      {
        embedder: googleAI.embedder('text-embedding-004'),
        collectionName: 'collectionName',
        clientParams: {
          url: 'http://localhost:6333',
        },
      },
    ]),
  ],
});
```

You'll need to specify a collection name, the embedding model you want to use and the Qdrant client parameters. In
addition, there are a few optional parameters:

- `embedderOptions`: Additional options to pass options to the embedder:

  ```js
  embedderOptions: { taskType: 'RETRIEVAL_DOCUMENT' },
  ```

- `contentPayloadKey`: Name of the payload filed with the document content. Defaults to "content".

  ```js
  contentPayloadKey: 'content';
  ```

- `metadataPayloadKey`: Name of the payload filed with the document metadata. Defaults to "metadata".

  ```js
  metadataPayloadKey: 'metadata';
  ```

- `dataTypePayloadKey`: Name of the payload filed with the document datatype. Defaults to "\_content_type".

  ```js
  dataTypePayloadKey: '_datatype';
  ```

- `collectionCreateOptions`: [Additional options](<(https://qdrant.tech/documentation/concepts/collections/#create-a-collection)>) when creating the Qdrant collection.

## Usage

Import retriever and indexer references like so:

```js
import { qdrantIndexerRef, qdrantRetrieverRef } from 'genkitx-qdrant';
```

Then, pass the references to `retrieve()` and `index()`:

```js
// To export an indexer:
export const qdrantIndexer = qdrantIndexerRef('collectionName', 'displayName');
```

```js
// To export a retriever:
export const qdrantRetriever = qdrantRetrieverRef(
  'collectionName',
  'displayName',
);
```

You can refer to [Retrieval-augmented generation](https://firebase.google.com/docs/genkit/rag) for a general discussion on indexers and retrievers.

### Score boosting / fusion

The retriever options accept optional `query` and `prefetch` fields, forwarded to the Qdrant [Query API](https://qdrant.tech/documentation/concepts/search/#query-api). When `query` is set, the embedded vector becomes a `prefetch` (overridable via `prefetch`) and `query` reranks the candidates, enabling formula boosting, fusion (RRF), and multi-stage prefetch. Omit both for a plain vector query.

```js
const docs = await ai.retrieve({
  retriever: qdrantRetriever,
  query: 'wireless headphones',
  options: {
    k: 20,
    // Boost in-stock items.
    query: { formula: { sum: ['$score', { mult: [0.2, 'in_stock'] }] } },
  },
});
```

### Grouped retrieval (diversity)

The retriever options accept an optional `groupBy` field (with optional `groupSize`), forwarded to the Qdrant [Grouping API](https://qdrant.tech/documentation/concepts/search/#grouping-api). When `groupBy` is set, results are grouped by that payload field and up to `groupSize` hits are returned per group (defaults to Qdrant's default of 3), across `k` groups. This diversifies results so a single over-represented group — e.g. one large document split into many chunks, or one dominant category — cannot crowd out the rest. Documents are returned as a flat list, group-ordered, with the group key exposed on each document's metadata as the reserved `_group` field (which overrides any same-named field in the document's own metadata). The `groupBy` field must be indexed. Note: re-sorting the returned documents by `_similarityScore` interleaves groups and undoes the diversity — keep the returned order to preserve grouping.

```js
const docs = await ai.retrieve({
  retriever: qdrantRetriever,
  query: 'fire-rated pipe penetration',
  options: {
    k: 8, // up to 8 groups
    groupBy: 'metadata.productFamily',
    groupSize: 3, // up to 3 chunks per family
    filter: {
      must: [{ key: 'metadata.wallType', match: { value: 'flexible' } }],
    },
  },
});
```
