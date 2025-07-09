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
                }
            }
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

- `dataTypePayloadKey`: Name of the payload filed with the document datatype. Defaults to "_content_type".

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
export const qdrantRetriever = qdrantRetrieverRef('collectionName', 'displayName');
```

You can refer to [Retrieval-augmented generation](https://firebase.google.com/docs/genkit/rag) for a general
discussion on indexers and retrievers.
