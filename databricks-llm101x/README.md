# Databricks LLM101x

## Large Language Models: Application through Production

[Databricks LLM101x](https://www.edx.org/course/large-language-models-101x) or
*Large Language Models: Application through production* is the application
focused, as opposed to theory focused, course of the two courses in the edX
Large Language Models Professional Certificate training.

## Quick Links

* [Module 1: Applications with LLMs](#module-1-applications-with-llms)
* [Module 2: Embeddings, Vector Databases, and Search](#module-2-embeddings-vector-databases-and-search)
* [Module 3: Multi-stage Reasoning](#module-3-multi-stage-reasoning)
* [Module 4: Fine-tuning and Evaluating LLMs](#module-4-fine-tuning-and-evaluating-llms)

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

Additional tasks that are general and overlap with other tasks:

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


## Module 3: Multi-stage Reasoning

* [Module 3 Resources](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@cef7038161874c9a8ee803140944e17a/block-v1:Databricks+LLM101x+2T2023+type@vertical+block@00b1f5f3a6cc47a592c5815cf378db8e)

### Prompt Engineering

Crafting more elaborate prompts to get the most out of our LLM interactions.

**Templating** - augments a known prompt that performs a specfic task well by
adding additional inofrmation as an input variable and standardizing the
output format so that the output of the LLM can be used as input to another
process such as a second LLM.

### LLM Chains

Linking multiple LLM interactions to build complexity and functionality.

**Multi-stage LLM chains** - for example, a sequentaial flow where an article
summary output from a summarization LLM feeds into a sentiment LLM.

### Agents

Giving LLMs the ability to delegate tasks to specified tools.

Agents are LLM-based systems that execute the ReasonAction loop.

* Thought -> Action -> Observation -> Repeat

To build an LLM we need:

* Task to be solveed or assigned to agents
* An LLM as the reasoning/decision making entity
* A set of tools that the LLM will select and execute to perform steps to
  achieve the task

## Module 4: Fine-tuning and Evaluating LLMs

* [Module 3 Resources](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@e57f7ee0973643318091f3b9f4a83911/block-v1:Databricks+LLM101x+2T2023+type@vertical+block@859b3255960844bba90a148ccb8fcac1)

Typical LLM Releases include.

Multiple:

* Sizes - S, M, L
* Sequence lengths (i.e. input tokens) - 512, 4,096, 62,000
* Flavors or tuned versions - base, chat, instruct

Tradeoffs between:

* Accuracy
* Speed
* Task-specific performance

### Applying Foundation LLMs

Task: Summarize news articles as riddles.

Potential Pipelines include routing data from the news API to:

* Few-shot open source LLM
* Open source instruction following LLM
* Paid LLM-as-a-service
* Build your own

### Fine-tuning: Few-shot Learning

**Pros**:

* speed of development
* performance
* cost - open source

**Cons**:

* data - need good-quality examples with context
* size-effect - may need to use largest model

Few-shot learning would likely need a long input sequence to create a proper
prompt as well as need the largest version of the LLM model to get adequate
performance

### Fine-tuning: Instruction-following LLMs

Useful for zero-shot learning.

**Pros**:

* data - no examples needed
* performance - if model fine-tuned out-of-box then should perform well
* cost - open source

**Cons**:

* quality - may be low if new if task doesn't meet training data
* size-effect - may need to use largest model

### Fine-tuning: LLM-as-a-service

**Pros**:

* speed of development - simple API call
* performance - vendor incentive to use large performant models

**Cons**:

* cost - not free
* data security/quality - data controlled by vendor
* vendor lock-in - dependent on vendor for model functionality

### Fine-tuning: DIY

Should be last case scenario as it will be very time and resource consuming.

**Pros**:

* task-tailoring
* inference cost
* control

**Cons**:

* time and compute cost
* data requirements
* skill-set dependencies

Need a model trained on open dataset or not commercially restricted like Dolly
or Dolly v2.

### Evaluating LLMs

* Training Loss/Validation Scores - only watched during training and not very
  helpful for end usage
* Perplexity - is the model surprised it got the right answer 
  - Good language model has high accuracy and low perplexity
  - Accuracy = next word is right or wrong
  - Perplexity = how confident was next word choice

### Task-specific Evaluations

N-Gram based

* Translation - BiLingual Evaluation Understudy (BLEU) 
* Summarization - ROUGE

Dataset Benchmarks

* Reading comprehension - Stanford Question Answering Dataset (SQuAD)

Alignment - which is used by OpenAI (helpful, honest, harmless)


### LLM Chain Evaluation

Guest lecture from Harrison Chase—creator of LangChain.

* Lack of data -  generate data, or wait and test as more data available
* Lack of metrics - visualize flow, use LLM as judge, or use user feedback

 Offline evaluation:

- Create dataset of test data points to run against
- Run chain or agent against them
- Visually inspect them
- Use LLM to auto-grade them

Online Evaluation:

* Direct feedback (thumbs up/down)
* Indirect feedback (clicked on link)
* Track feedback over time

## Module 5: Fine-tuning and Evaluating LLMs


