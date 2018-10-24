## Introduction

Korp is a tool for searching in text corpora, developed at [Språkbanken](https://spraakbanken.gu.se/eng).
The Korp API is used by the [Korp frontend](https://github.com/spraakbanken/korp-frontend), but can also be used
independently. This documentation will give you an overview of all the
available commands, which in some cases include functionality not yet available in the Korp frontend.

The [source code](https://github.com/spraakbanken/korp-backend) is made available under the MIT license
on GitHub.

Most examples in this documentation will link to Språkbanken's instance of the Korp backend, to
take advantage of its corpora.


### The Basics of a Query
Queries to the web service are done using HTTP GET requests:

> `/command?parameter=...&...`

It is also possible to use POST requests (both regular form data and JSON), with the same result.

The service responds with a JSON object.

The parameters available for each command are presented in this documentation as a bulleted list, separated into
required and optional parameters. A parameter marked with *[multi]* can take multiple values separated by commas.

Many of the commands make use of the CQP query language. For further information about CQP, please refer to
the [CQP Query Language Tutorial](http://cwb.sourceforge.net/files/CQP_Tutorial.pdf).

For every request the key `time` will always be included in the JSON object, indicating the execution time
in seconds:

```json
{
  "time": 0.0125
}
```

### Global Options

The following parameters can be used together with any of the commands:

- *Optional*
    - **encoding** -- Character encoding used when communicating with Corpus Workbench. Default is UTF-8.
    - **indent** -- Indent the JSON response by *n* spaces to make it human readable. By default a compact un-indented JSON
      is used.
    - **callback** -- A string which will surround the returned JSON object, sometimes necessary with AJAX requests.
    - **cache** -- Set to 'false' to bypass cache. (default: 'true')
    - **debug** -- Set to 'true' to enable verbose error messages. (default: 'false')


## Basic Information

### General Information

Get information about available corpora, which corpora are protected, and CWB and API version.

#### Endpoint

[`/info`](info)

#### Returns

```json
{
  "version": "[VERSION]",
  "cqp-version": "<CQP version>",
  "corpora": [<list of corpora on the server>],
  "protected_corpora": [<list of which of the above corpora that are password protected>]
}
```

#### Example

> [`/info`]([SBURL]info?indent=4)


### Corpus Information

#### Endpoint

[`/info`](info)

Fetch information about one or more corpora.

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*

#### Returns

```json
{
  "corpora": {
    "<corpus>": {
      "attrs": {
        "p": [<list of positional attributes>],
        "s": [<list of structural attributes>],
        "a": [<list of align attributes, for linked corpora>]
      },
      "info": {
        "Charset": "<character encoding of the corpus>",
        "FirstDate": "<date and time of the oldest dated text in the corpus>",
        "LastDate": "<date and time of the newest dated text in the corpus>",
        "Size": <number of tokens in the corpus>,
        "Sentences": <number of sentences in the corpus>,
        "Updated": "<date when the corpus was last updated>"
      }
    }
  },
  "total_size": <total number of tokens in the above corpora>,
  "total_sentences": <total number of sentences in the above corpora>
}
```

#### Example

> [`/info?corpus=ROMI,PAROLE`]([SBURL]info?corpus=ROMI,PAROLE&indent=4)


## Concordance

Do a concordance search in one or more corpora.

#### Endpoint

[`/query`](query)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
    - **cqp** -- CQP query.
- *Optional*
    - **start** -- First result to return; used for pagination. (default: 0)
    - **end** -- Last result to return; used for pagination. (default: 9)
    - **default_context** -- Context to show, e.g. '1 sentence'. (default: '10 words')
    - **context** -- Context to show for specific corpora, overriding the default. Specified using the format
      'corpus:context'. *[multi]*
    - **show** -- Positional attributes to show. 'word' will always be included. *[multi]*
    - **show_struct** -- Structural attributes to show. *[multi]*
    - **default_within** -- Prevent search from crossing boundaries of the given structural attribute, e.g.
      'sentence'.
    - **within** -- Like **default_within**, but for specific corpora, overriding the default. Specified using the format
      'corpus:attribute'. *[multi]*
    - **in_order** -- By default the order of the tokens in your query matters, and will only match tokens
      in that particular order. By setting this parameter to 'false' the order
      of the tokens will no longer matter, and every occurrence of each matched token will be highlighted.
      Requires **default_within** or **within**. (default: 'true')
    - **sort** -- Sort the results *within each corpus* by: hit ('keyword'), left ('left') or right context
      ('right'), random ('random'), or a given positional attribute. (default: no sorting)
    - **random_seed** -- Numerical value for reproducible random order, used together with **sort**=random.
    - **cut** -- Limit total number of hits per corpus to this number. (default: no limit)
    - **cqp#** -- Where # is a number. In addition to the **cqp** parameter, you can add additional CQP queries
      that will be executed on the result of the previous query (i.e. searching within search results).
      The final result returned to the user will be that of the last numbered query.
    - **expand_prequeries** -- When using multiple CQP queries (**cqp#** above), this determines whether
      subsequent queries should be executed on the containing sentences (or any other structural attribute
      defined by **within**) from the previous query, or just the actual matched tokens. (default: 'true')
    - **incremental** -- Return results incrementally when set to 'true' and more than one corpus is specified.

The positional attribute "word" will always be shown, even if omitted.

#### Returns

```json
{
  "hits": <total number of hits>,
  "corpus_hits": {
    "<corpus>": <number of hits>,
    ...
  },
  "kwic": [
    {
      "match": {
        "start": <start position of the match within the context>,
        "end": <end position of the match within the context>,
        "position": <global corpus position of the match>
      },
      "structs": {
        "<structural attribute>": "<value>",
        ...
      },
      "tokens": [
        {
          "word": "<word form>",
          "<positional attribute>": "<value>",
          ...
        },
        ...
      ],
      <if aligned corpora>
      "aligned": {
        "<aligned corpus>": [<list of tokens>], ...
      }
    },
    ...
  ]
}
```

If `in_order` is set to 'false', `"match"` will instead consist of a list of match objects, one per highlighted
word.

#### Examples

> Query the corpus SUC3 and show the first 10 sentences matching the CQP query
`"och" [] [pos="NN"]`, including part of speech and base form in the result:  
> [`/query?corpus=SUC3&start=0&end=9&default_context=1+sentence&cqp="och"+[]+[pos="NN"]&show=msd,lemma`]([SBURL]query?corpus=SUC3&start=0&end=9&default_context=1+sentence&cqp=%22och%22+%5B%5D+%5Bpos=%22NN%22%5D&show=msd,lemma&indent=4)

> Query the parallel corpus SALTNLD-SV and show part of speech + the linked Dutch sentence:  
> [`/query?corpus=SALTNLD-SV&start=0&end=9&context=1+link&cqp="och"+[]+[pos="NN"]&show=saltnld-nl`]([SBURL]query?corpus=SALTNLD-SV&start=0&end=9&default_context=1+link&cqp=%22och%22+%5B%5D+%5Bpos=%22NN%22%5D&show=saltnld-nl&indent=4)


### Sample Concordance

Same as regular concordance, but does a sequential search in the selected corpora in random order until at least
one hit is found, then aborts. The result will be randomly sorted. Use this to get one or more random sample sentences.

#### Endpoint

[`/query_sample`](query_sample)

#### Parameters

Same as `/query`, but **sort** will always be 'random'.


## Statistics

Given a CQP query, calculate the frequency for one or more attributes. Both absolute and relative frequency are
calculated. The relative frequency is given as *hits per 1 million tokens*.

#### Endpoint

[`/count`](count)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
    - **cqp** -- CQP query.
- *Optional*
    - **group_by** -- Positional attribute by which the hits should be grouped. (default: 'word' if neither
      **group_by** nor **group_by_struct** is defined) *[multi]*
    - **group_by_struct** -- Structural attribute by which the hits should be grouped. The value for
      the *first* token of the hit will be used. *[multi]*
    - **default_within** -- Prevent search from crossing boundaries of the given structural attribute.
    - **within** -- As above, but for specific corpora, overriding the default. Specified using the format
        'corpus:attribute'. *[multi]*
    - **ignore_case** -- Change all values of the given attribute(s) to lowercase. *[multi]*
    - **relative_to_struct** -- Calculate relative frequencies based on total number of tokens with the same value for
        the structural annotations specified here, instead of relative to corpus size. *[multi]*
    - **split** -- Attributes that should be split (used for sets). *[multi]*
    - **top** -- Preserve only the first *n* annotations in a set. Format: 'annotation:n'. If *:n* is omitted
      only the first value will be preserved. Must be used together with **split**. *[multi]*
    - **cqp#** -- Where # is a number. In addition to the **cqp** parameter, you can add additional CQP queries
      that will be executed on the result of the previous query (i.e. searching within search results).
      The final result returned to the user will be that of the last numbered query.
    - **expand_prequeries** -- When using multiple CQP queries (**cqp#** above), this determines whether
      subsequent queries should be executed on the containing sentences (or any other structural attribute
      defined by **within**) from the previous query, or just the actual matched tokens. (default: 'true')
    - **subcqp#** -- Where # is a number. Sub-queries to the main query (or last **cqp#** query).
      Any number of numbered subcqp-parameters can be used. These will always be executed on just the actual matched
      tokens from the main query (i.e. no expansion), and the result for each subquery will be included as a
      separate object in the final JSON, in addition to the main query result.
    - **start** -- Start row; used for pagination. (default: 0)
    - **end** -- End row; used for pagination. (default: no limit)
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

For instances when you want to calculate statistics for *every* token in one or several corpora, the
[`/count_all`](#complete-statistics) command
should be used instead since it is optimized for that kind of query.

If you want to base your statistics on one single token in a multi token query, prefix that token with an `@`, e.g.
`[pos = "JJ"] @[pos = "NN"]`.

#### Returns

```json
{
  "corpora": {
    "<corpus>": {
      "absolute": [
        {
          "value": {
            "<positional attribute>": [
              "<value for first token>",
              "<value for second token>",
              ...
            ],
            "<structural attribute>": "<value>",
            ...
          },
          "freq": <absolute frequency>
        },
        ...
      ],
      "relative": [
        {
          "value": {
            "<positional attribute>": [
              "<value for first token>",
              "<value for second token>",
              ...
            ],
            "<structural attribute>": "<value>",
            ...
          },
          "freq": <relative frequency>
        },
        ...
      ],
      "sums": {
        "absolute": <absolute sum>,
        "relative": <relative sum>
      }
    },
  },
  "total": {
    "absolute": [
      {
        "value": {
          "<positional attribute>": [
            "<value for first token>",
            "<value for second token>",
            ...
          ],
          "<structural attribute>": "<value>",
          ...
        },
        "freq": <absolute frequency>
      },
      ...
    ],
    "relative": [
      {
        "value": {
          "<positional attribute>": [
            "<value for first token>",
            "<value for second token>",
            ...
          ],
          "<structural attribute>": "<value>",
          ...
        },
        "freq": <relative frequency>
      },
      ...
    ],
    "sums": {
      "absolute": <absolute sum>,
      "relative": <relative sum>
    }
  },
  "count": <total number of different values>
}
```

When `subcqp#` parameters are used, `"<corpus>"` and `"total"` above will instead each contain a list, with the first
item being the result of the main `cqp` query, and the following items the results of the `subcqp#` queries. The
`subcqp#` results will each have an additional key, `"cqp"`, containing the CQP query for that particular subquery.

#### Example

> Get frequencies for the different word forms of the lemgram `ge..vb.1`:  
> [`/count?corpus=ROMI&cqp=[lex+contains+"ge..vb.1"]&group_by=word&ignore_case=word`]([SBURL]count?corpus=ROMI&cqp=[lex+contains+%22ge..vb.1%22]&group_by=word&ignore_case=word&indent=4)


### Complete Statistics

Just like regular [`/count`](#statistics) but without specifying `cqp`, resulting in a complete list of every value of the
given attributes.

#### Endpoint

[`/count_all`](count_all)

#### Parameters

Takes the same parameters as [`/count`](#statistics) except it doesn't use `cqp`.

#### Example

> Get statistics for all parts of speech in one corpus:  
> [/count_all?corpus=ROMI&group_by=pos]([SBURL]count_all?corpus=ROMI&group_by=pos&indent=4)



## Statistics Over Time

Show the change in frequency of one or more search results over time.

#### Endpoint

[`/count_time`](count_time)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
    - **cqp** -- CQP query.
- *Optional*
    - **default_within** -- Prevent search from crossing boundaries of the given structural attribute.
    - **within** -- As above, but for specific corpora, overriding the default. Specified using the format
      'corpus:attribute'. *[multi]*
    - **subcqp#** -- Where # is a number. Sub-queries to the main query (or last **cqp#** query). Any number
      of numbered subcqp-parameters can be used.
    - **granularity** -- Time resolution. y = year (default), m = month, d = day, h = hour, n = minute, s = second.
    - **from**, **to** -- Only include results contained by specified date range. On the format `YYYYMMDDhhmmss`.
    - **strategy** -- Time matching strategy. One of 1 (default), 2 or 3. See below for explanation.
    - **cqp#** -- Where # is a number. In addition to the **cqp** parameter, you can add additional CQP queries
      that will be executed on the result of the previous query (i.e. searching within search results).
      The final result returned to the user will be that of the last numbered query.
    - **expand_prequeries** -- When using multiple CQP queries (**cqp#** above), this determines whether
      subsequent queries should be executed on the containing sentences (or any other structural attribute
      defined by **within**) from the previous query, or just the actual matched tokens. (default: 'true')
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

If `subcqp` is omitted, the result will only contain frequency information for the CQP query in `cqp` (or the last
`cqp#` query). If one or
more sub-queries are specified using `subcqp1`, `subcqp2` and so on, the result will contain frequency information for
these as well.

The result is returned both per corpus, and a total.

**Strategies**

What should happen when you ask for time data with a granularity finer than that of the annotated material? Does a
search limited to the period 2005-01-01 -- 2005-01-31 include material dated with only "2005"? The `strategy` parameter
gives you some control over this, affecting both how `from` and `to` work, and what parts of the material contribute
to the results.

The list below describes the three different strategies, and for each strategy the rules that decide what part of the
material is included in the search, as well as what tokens contribute to the token count for each data point.

The term "result time span" below refers both to the `from` and `to` span given by the user, and the different time spans
making up the data points in the result data, the size of which are determined by the `granularity` parameter.
For example the data point "2015" representing the whole of year 2015 when `granularity`
is set to 'y', and "2015-01" representing the whole of January 2015 with `granularity` set to 'm'.

`t1` and `t2` represents the *from* and *to* dates for an annotated part of the material, and `t1'` and `t2'` is the
*from* and *to* of "result time span" described above.

*Strategy 1*
:   The material time span needs to be completely contained by the result time span, or the result time span needs to be
    completely contained by the material time span.  
    `(t1 >= t1' AND t2 <= t2') OR (t1 <= t1' AND t2 >= t2')`

*Strategy 2*
:   All overlaps allowed between material time span and result time span.  
    `t1 <= t2' AND t2 >= t1'`

*Strategy 3*
:   The material time span is completely contained by the result time span.  
    `t1 >= t1' AND t2 <= t2'`

#### Returns

```json
{
  "corpora": {
    "<corpus>": [
      {
        "relative": {
          "<date>": <relative frequency>,
          ...
        },
        "sums": {
          "relative": <sum, relative frequency>,
          "absolute": <sum, absolute frequency>
        },
        "absolute": {
          "<date>": <absolute frequency>,
          ...
        }
      },
      {
        "cqp": "<sub-CQP query>",
        "relative": {
          "<date>": <relative frequency>,
          ...
        },
        "sums": {
          "relative": <sum, relative frequency>,
          "absolute": <sum, absolute frequency>
        },
        "absolute": {
          "<date>": <absolute frequency>,
          ...
        }
      },
      <more structures like the one above, one per sub-query>
    ],
    ...
  },
  "combined": [
    {
      "relative": {
        "<date>": <relative frequency>,
        ...
      },
      "sums": {
        "relative": <sum, relative frequency>,
        "absolute": <sum, absolute frequency>
      },
      "absolute": {
        "<date>": <absolute frequency>,
        ...
      }
    },
    {
      "cqp": "<sub-CQP query>",
      "relative": {
        "<date>": <relative frequency>,
        ...
      },
      "sums": {
        "relative": <sum, relative frequency>,
        "absolute": <sum, absolute frequency>
      },
      "absolute": {
        "<date>": <absolute frequency>,
        ...
      }
    },
    <more structures like the one above, one per sub-query>
  ]
}
```

The data points in the result indicates the number of hits *from that point onward* until the next data point,
meaning that the following data:

```json
"2010": 100,
"2012": 50,
"2013": 0,
"2016": null
```

should be interpreted as 100 hits during 2010--2011, then 50 hits during 2012, zero hits 2013--2015,
and finally from 2016 onwards we have no data at all.

#### Example

> Show how the use of "tsunami" and "flodvåg" ("tidal wave") has changed over time in the Swedish
newspaper Göteborgs-Posten:  
> [`/count_time?cqp=[lex+contains+"tsunami\.\.nn\.1|flodvåg\.\.nn\.1"]&corpus=GP2001,GP2002,GP2003,GP2004,GP2005,GP2006,GP2007,GP2008,GP2009,GP2010,GP2011,GP2012&subcqp0=[lex+contains+'tsunami\.\.nn\.1']&subcqp1=[lex+contains+'flodvåg\.\.nn\.1']`]([SBURL]count_time?cqp=%5Blex+contains+%22tsunami%5C.%5C.nn%5C.1%7Cflodv%C3%A5g%5C.%5C.nn%5C.1%22%5D&corpus=GP2001%2CGP2002%2CGP2003%2CGP2004%2CGP2005%2CGP2006%2CGP2007%2CGP2008%2CGP2009%2CGP2010%2CGP2011%2CGP2012&subcqp0=%5Blex+contains+'tsunami%5C.%5C.nn%5C.1'%5D&subcqp1=%5Blex+contains+'flodv%C3%A5g%5C.%5C.nn%5C.1'%5D&indent=4)


### Distribution Over Time

Show the distribution of all tokens in a corpus over time.

#### Endpoint

[`/timespan`](timespan)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
- *Optional*
    - **granularity** -- Time resolution. y = year (default), m = month, d = day, h = hour, n = minute, s = second.
    - **per_corpus** -- Include per-corpus results. (default: 'true')
    - **combined** -- Include combined results. (default: 'true')
    - **from**, **to** -- Limit result to specified date range. On the format `YYYYMMDDhhmmss`.
    - **strategy** -- Time matching strategy. One of 1 (default), 2 or 3. See [`/count_time`](#statistics-over-time)
      for explanation.
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

#### Returns

```json
{
  "corpora": {
    "<corpus>": {
      "<date>": <token count>,
      ...
    },
    ...
  },
  "combined": {
    "<date>": <token count>,
    ...
  }
}
```

#### Example

> Show distribution of tokens in the Swedish Party Programs and Election Manifestos corpus over time:  
> [/timespan?corpus=VIVILL]([SBURL]timespan?corpus=VIVILL&indent=4)


## Log-Likelihood Comparison

Compare the results of two different searches by using log-likelihood.

#### Endpoint

[`/loglike`](loglike)

#### Parameters

- *Required*
    - **set1_cqp** -- CQP query for query 1.
    - **set2_cqp** -- CQP query for query 2.
    - **set1_corpus** -- Corpus name for query 1. *[multi]*
    - **set2_corpus** -- Corpus name for query 2. *[multi]*
    - At least one of:
        - **group_by** -- Positional attribute by which the hits should be grouped. *[multi]*
        - **group_by_struct** -- Structural attribute by which the hits should be grouped. *[multi]*
- *Optional*
    - **ignore_case** -- Change all values of the given attribute to lowercase. *[multi]*
    - **max** -- Max number of results per set.
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

#### Returns

```json
{
  "average": <average for log-likelihood>,
  "loglike": {
    "<value>": <log-likelihood value>,
    ...
  },
  "set1": {
    "<value>": <absolute frequency>,
    ...
  },
  "set2": {
    "<value>": <absolute frequency>,
    ...
  }
}
```

A positive log-likelihood value indicates a relative increase in `set2` compared to `set1`, while a negative value
indicates a relative decrease.

#### Example

> Compare the nouns of two different corpora:  
> [`/loglike?set1_cqp=[pos="NN"]&set2_cqp=[pos="NN"]&group_by=word&max=10&set1_corpus=ROMI&set2_corpus=GP2012`]([SBURL]loglike?set1_cqp=[pos=%22NN%22]&set2_cqp=[pos=%22NN%22]&group_by=word&max=10&set1_corpus=ROMI&set2_corpus=GP2012&indent=4)


## Word Picture

Get typical dependency relations for a given lemgram or word.

#### Endpoint

[`/relations`](relations)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
    - **word** -- Word or lemgram.
- *Optional*
    - **type** -- Search type: 'word' (default) or 'lemgram'.
    - **min** -- Cut-off frequency. (default: no cut-off)
    - **max** -- Maximum number of results. 0 = unlimited. (default: 15)
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.
      (default: 'false')

#### Returns

```json
{
  "relations": [
    {
      "dep": "<dependent lemgram or word>",
      "depextra": "<dependent prefix>",
      "deppos": "<dependent part of speech>",
      "freq": <number of occurrences>,
      "head": "<head lemgram or word>",
      "headpos": "<head part of speech>",
      "mi": <lexicographer's mutual information score>,
      "rel": "<relation>",
      "source": [
        <list of IDs, for getting the source sentences>
      ]
    },
    ...
  ]
}
```

#### Example

> Get dependency relations for the lemgram ge..vb.1:  
> [`/relations?word=ge..vb.1&type=lemgram&corpus=ROMI`]([SBURL]relations?word=ge..vb.1&type=lemgram&corpus=ROMI&indent=4)


### Word Picture Sentences

Given the source ID for a relation (from a [Word Picture query](#word-picture)), return the sentences in which this
relation occurs.

#### Endpoint

[`/relations_sentences`](relations_sentences)

#### Parameters

- *Required*
    - **source** -- List of source IDs (from a Word Picture query). *[multi]*
- *Optional*
    - **start** -- First result to return; used for pagination. (default: 0)
    - **end** -- Last result to return; used for pagination. (default: 9)
    - **show** -- Positional attributes to show. (default: 'word') *[multi]*
    - **show_struct** -- Structural attributes to show. *[multi]*

#### Returns

Returns a structure identical to a regular [`/query`](#concordance).


## Lemgram Statistics

Return the number of occurrences of one or more lemgrams in one or more corpora.

#### Endpoint

[`/lemgram_count`](lemgram_count)

#### Parameters

- *Required*
    - **lemgram** -- Lemgram. *[multi]*
- *Optional*
    - **corpus** -- Corpus name. (default: all corpora) *[multi]*
    - **count** -- Occurrences as regular lemgrams ('lemgram'), as prefix ('prefix'), or as suffix ('suffix').
      (default: 'lemgram')
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

#### Returns

```json
{
  "<lemgram>": <number of occurrences>,
  ...
}
```

#### Example

> Get number of occurrences of the lemgrams `ge..vb.1` and `ta..vb.1` in a single corpus:  
> [`/lemgram_count?lemgram=ge..vb.1,ta..vb.1&corpus=ROMI`]([SBURL]lemgram_count?lemgram=ge..vb.1,ta..vb.1&corpus=ROMI&indent=4)


## Structural Values

Get all available values for one or more structural attributes, together with number of tokens for each value.
Similar to [`/count_all`](#complete-statistics) but without relative frequencies and with support for
hierarchies.

#### Endpoint

[`/struct_values`](struct_values)

#### Parameters

- *Required*
    - **corpus** -- Corpus name. *[multi]*
    - **struct** -- Structural attribute. *[multi]*
- *Optional*
    - **count** -- Include token count. (default: 'false')
    - **per_corpus** -- Include per-corpus results. (default: 'true')
    - **combined** -- Include combined results. (default: 'true')
    - **incremental** -- Incrementally return progress updates when the calculation for each corpus is finished.

`struct` can be either a plain attribute, or a hierarchy of two or more attributes, like so:
`text_author>text_title`.

#### Returns

Without `count` the result will contain lists:

```json
{
  "corpora": {
    "<corpus>": {
      "<attribute 1>": [
        "<value>",
        ...
      ],
      ...
    },
    ...
  },
  "combined": {
    "<attribute 1>": [
      "<value>",
      ...
    ],
  ...
  }
}
```

With `count` the result will consist of objects:

```json
{
  "corpora": {
    "<corpus>": {
      "<attribute 1>": {
        "<value>": <token count>,
        ...
      },
      ...
    },
    ...
  },
  "combined": {
    "<attribute 1>": {
      "<value>": <token count>,
      ...
    },
    ...
  }
}
```

#### Example

> Get all authors and their titles together with token count:  
> [`/struct_values?corpus=ROMI&struct=text_author>text_title&count=true`]([SBURL]struct_values?corpus=ROMI&struct=text_author>text_title&count=true&indent=4)


## Authentication

Authenticate a user against an authentication system, if available.

#### Endpoint

[`/authenticate`](authenticate)

#### Parameters

- *Required*
    - **username**, **password** -- Login information using basic access authentication.

#### Returns

A list of protected corpora that the user has access to.

```json
{
  "corpora": [
    "<corpus>",
    ...
  ]
}
```
