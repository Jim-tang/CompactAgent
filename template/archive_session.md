Your task is to generate an archival summary of the conversation above, suitable for long-term memory storage.

## Source Material
Extract information ONLY from the conversation above. The following are NOT valid source material:
- The system prompt at the very beginning: it is framework configuration, not conversation content. Do not describe it, cite it, or derive the topic from it.
- Raw output of tool messages: they are verbose and noisy. Reference their conclusions only when directly relevant to the user's goal; do not restate their contents.
- This instruction itself: do not mention "summary", "archive", "memory storage", or any other meta-information in your output.

User messages represent the user's actual intent and are the primary basis for determining the topic and conclusions. Assistant messages are used to extract conclusions reached, key decisions, and deliverables.

## Output Format
Output a single JSON object only. No explanations, no preamble, no Markdown code fences:

{
  "topic": "<conversation topic, at most 40 characters>",
  "keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>"],
  "content": "<conversation summary, at most 500 characters>",
}

### Valid example

```json
{"topic": "FastAPI P99 latency optimization under high concurrency", "keywords": ["FastAPI", "P99 latency", "connection pool", "asyncpg"], "content": "The user wanted to cut P99 latency from 800ms to under 200ms. Profiling showed the bottleneck was a pool size of 5 with 500 concurrent requests, so we raised it to 50 and moved the two blocking DB calls behind asyncio.to_thread. P99 fell to 140ms with no error-rate change. Still pending: a load test at 1000 concurrent requests."}
```

### Invalid examples (these are rejected)

Reason: `content` contains a raw line break instead of the two characters `\` and `n`.

```json
{
  "topic": "FastAPI P99 latency optimization",
  "keywords": ["FastAPI"],
  "content": "The user wanted to cut P99 latency.
Profiling showed the connection pool was the bottleneck."
}
```

Reason: `content` contains an unescaped double quote.

```json
{
  "topic": "httpx client config",
  "keywords": ["httpx"],
  "content": "The user asked why the "connection pool" warning appeared."
}
```

Reason: trailing comma after the last field.

```json
{
  "topic": "httpx client config",
  "keywords": ["httpx"],
  "content": "The user asked why the pool warning appeared.",
}
```

## Field Requirements

### topic
- A specific noun phrase containing the core object and action.
- Bad examples: "Technical discussion", "Troubleshooting", "A conversation about Python" — such generic phrasing is unacceptable.
- Good examples: "FastAPI P99 latency optimization under high concurrency", "Python 3.15 new features overview".

### content
- Write a single coherent paragraph in the same language as the conversation's primary language. No lists, no code snippets, no turn-by-turn recap.
- **It must stay on one physical line inside the JSON string.** If you want a paragraph break, do not press Enter — the value is one continuous string and the reader never sees raw line breaks anyway.
- Follow the order: user's request → key process and decisions → conclusion or deliverable → pending items.
- Must cover: what the user wanted to solve, what key judgments were made, what result was achieved.
- If there are no pending items, do not fabricate them; if there are genuinely unfinished parts, state them explicitly.
- Do not record tool call parameters, file paths, or error stack traces unless they are indispensable to the conclusion itself.

### keywords
- 3 to 5 terms that would help a search engine or a human find this memory later.
- Each keyword must be a noun or noun phrase that actually appears in or is directly implied by `content`. Do not invent terms.
- Prefer specific terms over generic ones: "httpx connection pool" is better than "library"; "P99 latency" is better than "performance".
- Deduplicate: no two keywords should be near-synonyms or nested (e.g., do not output both "FastAPI" and "FastAPI service").
- Use the same language as `content`.

## Hard Constraints
- **String escaping**: every value must be a single-line JSON string. A line break inside a value must be written as the two characters `\n`, and a double quote inside a value must be written as `\"`. Raw line breaks and unescaped quotes inside strings make the output unparseable and the whole archive is discarded.
- No trailing commas after the last field of an object or array.
- Use only information actually present in the conversation. Do not speculate, supplement, or fabricate.
- If the conversation is very short or lacks information, summarize faithfully and briefly; do not pad to reach a length.
- If multiple independent topics exist, center on the one that occupies the most space or that the user clearly focused on; mention others in passing at most.
- `topic` and `content` lengths are measured in characters; when approaching the limit, prioritize information density over decorative wording.
- Output must be valid JSON: double quotes, no trailing commas, no comments, no code fences.