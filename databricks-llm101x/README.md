# Databricks LLM101x

## Large Language Models: Application through Production

[Databricks LLM101x](https://www.edx.org/course/large-language-models-101x) or
*Large Language Models: Application through production* is the application
focused, as opposed to theory focused, course of the two courses in the edX
Large Language Models Professional Certificate training.

## Module 1: Applications with LLMs

* [Module 1 Resources](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@0ff664ca2af041ffa79292147229cbd5/block-v1:Databricks+LLM101x+2T2023+type@vertical+block@ff64e77cc19c48c288d89b2428686710?_gl=1%2A1mtzra4%2A_ga%2AMTI1NTAxNzMwMS4xNjgzMzg3NzYw%2A_ga_D3KS4KMDT0%2AMTY4Njc1ODYxMS4zNi4xLjE2ODY3NTg2MjAuNTEuMC4w)

### Hugging Face

The course starts off with a brief introduction to the Hugging Face. But I think
the official documentation is a better resource for reference.

[*Hugging Face Documentation*](https://huggingface.co/docs)

### Model Selection

Considerations for selecting the right model to begin development inlcud:

* Model resource requirements
* Fine tuned training
* Previous usage and documentation
* Task specific performance

***Model Discovery***

* [Stanford Ecosystem Graphs](https://crfm.stanford.edu/ecosystem-graphs/)

### NLP Tasks

#### Common NLP Tasks

Covered in course:

* Summarization - shortening the text while preserving key information
* Sentiment Analysis - labeling text as positive or negative with a score
* Translation - converting text from one language to another
* Zero-shot classification - classifying text without re-training on the labels
* Few-shot learning - train a model to do a new task with a few training
  examples of that task

Addtional tasks that are general and overlap with other tasks:

* Conversation or chat - generating text in response to a prompt
* (Table) question-answering - answering questions about a table
* Text / token classification - classifying each token in a sentence
* Text generation - generating text from scratch

### Prompts

Instruction-following LLMS are trained to generate text from an arbitrary
instruction or prompt.

Prompts are inputs or queries to the model to elicit responses. Prompts can be
natural language sentences, code snippets or any combination thereof--basically
any text.

Additionally, prompts can be outputs from other LLM queries. This allows nesting
or chaining LLMs to create complex and dynamic interactions.

### Prompt Engineering

Prompt engineering is model specific and different use cases for the same model
may require different prompts.

Prompts guide a model to complete a task and good prompts should be clear and
specific.

Good prompts typically include:

* Instruction
* Context
* Input or question
* Output type or format

**Make sure to iterate on the prompt to get the best results**

#### Helping The Model

* Ask the model not to make things up or hallucinate
* Ask the model not to assume or probe for sensitive information
* Ask the model not to rush to a solution
  * 'Explain how you solved this problem'  
  * 'Do this step-by-step'

#### Prompt Formatting

Use delimiters to distinguish between instruction, context, input and output.

* Pound signs ###
* Backtikcs ```
* Braces {} or brackets []
* Dashes ---

Ask the model to return structured output.

* JSON, XML, HTML, Markdown, etc...

Provide a correct example with formatting.

#### Prompt Hacking

Good prompts reduce successful prompt hacking or exploiting LLM vulnerabilities
by manipulating inputs.

* Prompt injection - adding additional text to the prompt
* Jailbreaking - bypassing a moderation rule or filter
* Prompt leaking - extracting sensitive information from the model

To avoid prompt hacking:

* Use post-processing or filter
* Repeat instructions in the prompt
* Enclose user input with random strings or tags
* Select a different model or restrict the prompt length

## Module 2: Embeddings, Vector Databases, and Search

* [Module 2 Resources](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@a7512e678c8a4fd98ae3635977fb55bb/block-v1:Databricks+LLM101x+2T2023+type@vertical+block@b475e8c705f448818785a4d5dc2952e4)

### Vector Search

* Two search strategies - exact (brute force) and approximate
* Distance between vectors is usually L1 (Manhattan) or most likely L2 (Euclidian)
* Cosine distance between vectors can be used to guage metric similarity 
* [Product Quantization](https://en.wikipedia.org/wiki/Vector_quantization) is often used for compression
* Algorithms 
  - [FAISS](https://ai.facebook.com/tools/faiss/) - L2 approximate search by
    areas or cohorts of nodes that works well for dense vector
  - [HNSW](https://github.com/nmslib/hnswlib) - L2 approximate search where vector nodes are put in layers and 

### Filtering

The categories of filtering strategies:

* Post-query - less performant becaus all data must be searched using brute
  force
* In-query - can be very compute intensive because more data must be loaded 
* Pre-query - can be very performant be eliminating data by filtering prior to
  similarity computation

### Vector Stores

Vector stores inlcudes bot vector databases and libraries.

Vector databases are similar to regular databases optimized for vectors. Their
major differentiator is that the database provides the combination of
enhanced vector search and traditional database properties like CRUD, ACID,
and big data scaling.

Vector libraries are ofent a much less complicated and easier implemented
option for proof of concept, research, and smaller data.

### Best Practices

The decision of whether a vector store is needed often comes down to
answering a single question.

> Do you need context augmentation for you LLM to provide it with additional
> knowledge?

Use cases where you often don't need augmentation.

* Summarization
* Text classification
* Translation

Improving retrieval.

* Embedding model selection
  - Do I have the right embedding model for my data?
  - Do my embeddings capture both my documents and queries?
* Document storage strategy
  - Should I store the whole document as one, or split it up into chunks?


