# Adding support for custom query languages in Korp

Query translation plugins let users search Korp using a language other than CQP. Each plugin declares which languages it
supports and provides a translator that converts queries into CQP expressions. Korp takes care of selecting the right
translator, validating its output, and running the resulting search.

For a complete working example, see `plugins/example_query_translation.py`.

## Creating a query translator

To add a new query language, implement the `QueryTranslator` protocol in a regular Python class. Your translator needs
two things:

* `languages`: A non-empty `frozenset[str]` containing the names of the languages your plugin supports. Names are
  case-sensitive, cannot have surrounding whitespace, and must be unique across enabled plugins. The name `"cqp"` is
  reserved.
* `async translate(ctx, language, query) -> str`: A method that converts a single query into a CQP expression. It
  receives the current request context, the language name, and the original query string.

The returned CQP expression must be non-empty and must not include a statement terminator (semicolon), named-query
assignment, or `within` clause. If a query cannot be represented as a single CQP expression, raise `ValueError` to
indicate that the construct is unsupported.

For example, a module named `plugins.my_translation` could look like this:

```py
from collections.abc import Mapping
from typing import Any

from korp.dependencies import Ctx


class WordTranslator:
    languages = frozenset({"literal-word"})

    async def translate(self, ctx: Ctx, language: str, query: str) -> str:
        return f'[word="{query}"]'


def create_translator(config: Mapping[str, Any]) -> WordTranslator:
    return WordTranslator()


QUERY_TRANSLATOR_FACTORY = create_translator
```

This simplified example assumes the query contains no characters that need escaping in CQP. A real translator should
handle escaping and other language-specific syntax as needed.

## Registering and configuring your plugin

Add your module to `PLUGINS`, for example:

```dotenv
PLUGINS = ["plugins.my_translation"]
```

Your module must export a synchronous callable named `QUERY_TRANSLATOR_FACTORY`. When the application starts, Korp calls
this factory once, passing it a deep copy of the module's `PLUGINS_CONFIG` settings (or an empty mapping if none are
provided).

Use the factory to check configuration, initialize any translation libraries, and set up annotation mappings. It should
return an instance of your translator. Invalid providers or duplicate language names prevent Korp from starting.

Korp registers translators by language in `app.state.query_translators`. The same translator instance may handle several
requests concurrently, so make sure to keep the parsing state local to each call.

If your translator performs potentially expensive synchronous parsing or compilation, run that work in a worker thread
to avoid blocking the event loop. For example:

```py
return await anyio.to_thread.run_sync(self._translate_sync, language, query)
```

## Testing your translator

You can test your translator without running a search using the `/query/translate` endpoint. Pass a `query` and
`query_language` parameter, either as a GET request with query parameters or a POST request with a JSON body. For
example:

```json
{"query": "cat", "query_language": "literal-word"}
```

Korp translates the query using your plugin and returns the resulting CQP expression:

```json
{"cqp": ["[word=\"cat\"]"]}
```

The endpoint is automatically available when at least one query translator is registered and also appears in the OpenAPI
specification and interactive API documentation.

## Using translated queries in searches

Once registered, your languages can be used with `/concordance`, `/concordance/sample`, `/frequencies`, and
`/frequencies/time`.

These endpoints accept either CQP queries through `cqp`, or queries in a registered language through `query` and
`query_language`.

For GET requests, repeat the parameters for multiple query stages. For POST requests, use JSON arrays, even when there
is only one stage:

```json
{"corpora": ["mycorpus"], "query": ["cat"], "query_language": ["literal-word"]}
```

You can specify one language for all query stages, or provide a separate language for each stage. When using multiple
languages, Korp pairs queries and languages by their positions in the respective lists.

For example:

```text
?query=FIRST&query=SECOND&query_language=grew&query_language=deptreepy
```

This translates `FIRST` using `grew` and `SECOND` using `deptreepy`. Order and duplicate entries are preserved.

Your translator only needs to translate one query at a time.

The caller is responsible for choosing suitable corpora, annotations, and search scope (`within`).

The `/log-likelihood` endpoint also supports translated queries. Each set independently accepts either `set1_cqp` or
`set1_query` with `set1_query_language`, and similarly for set 2. Unlike the search endpoints, these fields contain
single strings rather than lists.

## Handling invalid and unsupported queries

Raise `ValueError` with a helpful message when a query cannot be translated. This includes invalid syntax, unsupported
language features, and queries that exceed your complexity limits.

Korp turns these errors into HTTP 422 responses with the code `invalid_request` and the appropriate query field. You can
also raise `APIValidationError` directly if you need to identify a specific request field.

Unknown query languages also produce HTTP 422 responses. Unexpected exceptions or invalid translator results are treated
as backend errors.
