import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from tqdm import tqdm

from FlagEmbedding import FlagReranker

import itertools

import pickle as pkl

#### CONFIG PARAMETERS ---

# Define the number of best sentences to pre-filter for with cosine similarity
NUM_SENTENCES_TO_CONSIDER = 40
# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        
        self.reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        
    def calculate_rankings(self, query, sentences):
        return self.reranker_model.compute_score(zip(itertools.repeat(query), sentences), normalize=True)

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any], retrieval_results) -> List[str]: # TODO: Can we have retrieval_results here? Maybe make it a class variable?
        # retrieval_results: Precomputed relevant chunks (sentences) for each query. This is a dictionary where the key is interaction_id, and the value is the list of relevant sentences.
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        query_times = batch["query_time"]
        batch_retrieval_results = [retrieval_results[i] for i in batch_interaction_ids] #maps each interaction_id to its corresponding precomputed relevant sentences from retrieval_results.
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers


    def precompute_relevance(self, dataloader):
        # Retrieve top matches for the whole batch
        retrieval_results = defaultdict(list)
        for batch in dataloader:
            batch_interaction_ids = batch["interaction_id"]
            queries = batch["query"]
            batch_search_results = batch["search_results"]
            query_times = batch["query_time"]

            # Chunk all search results using ChunkExtractor
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                batch_interaction_ids, batch_search_results
            )

            # Calculate all chunk embeddings
            chunk_embeddings = self.calculate_embeddings(chunks)

            # Calculate embeddings for queries
            query_embeddings = self.calculate_embeddings(queries)

            for _idx, interaction_id in enumerate(batch_interaction_ids):
                query = queries[_idx]
                query_time = query_times[_idx]
                query_embedding = query_embeddings[_idx]

                # Identify chunks that belong to this interaction_id
                relevant_chunks_mask = chunk_interaction_ids == interaction_id

                # Filter out the said chunks and corresponding embeddings
                relevant_chunks = chunks[relevant_chunks_mask]
                relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

                # Calculate cosine similarity between query and chunk embeddings,
                cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

                # and retrieve top-N results.
                cosine_results = relevant_chunks[
                    (-cosine_scores).argsort()[:NUM_SENTENCES_TO_CONSIDER]
                ]
                
                scored_cosine_results = self.calculate_rankings(query, cosine_results)
                
                batch_retrieval_results = cosine_results[
                    (-scored_cosine_results).argsort()[:NUM_CONTEXT_SENTENCES]
                ]
                
                # You might also choose to skip the steps above and 
                # use a vectorDB directly.
                retrieval_results[interaction_id].append(batch_retrieval_results)

        with open('retrieval_results.pkl', 'wb') as f:
            pkl.dump(f, retrieval_results)
            

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts

### EXAMPLE
        
# if __name__ == "__main__":

#     ##EXAMPLE
    
#     dataloader = [
#         {
#             "interaction_id": ["query_1", "query_2"],
#             "query": ["What is machine learning?", "Explain the capital of France."],
#             "search_results": [
#                 [
#                     {"page_result": "<html><body>Machine learning is a subset of AI...</body></html>"},
#                     {"page_result": "<html><body>It focuses on algorithms and patterns...</body></html>"},
#                 ],
#                 [
#                     {"page_result": "<html><body>The capital of France is Paris...</body></html>"},
#                     {"page_result": "<html><body>Paris is known as the city of lights...</body></html>"},
#                 ],
#             ],
#             "query_time": ["2024-12-01T12:00:00Z", "2024-12-01T12:05:00Z"],
#         }
#     ]


#     rag_model = RAGModel()
#     rag_model.precompute_relevance(dataloader)

#     #file will have precomputed relevance information
#     with open('retrieval_results.pkl', 'rb') as f:
#         retrieval_results = pkl.load(f) #retrieve the retrieval_results dictionary which maps interaction_id (query IDs) to the precomputed top-ranked sentences (chunks).

#     print("\nPrecomputed Relevance Results:")
#     for interaction_id, relevant_chunks in retrieval_results.items():
#         print(f"Interaction ID: {interaction_id}")
#         for idx, chunk in enumerate(relevant_chunks, 1):
#             print(f"  Chunk {idx}: {chunk}")

#     # Example batch for answer generation
#     batch = {
#         "interaction_id": ["query_1", "query_2"],
#         "query": ["What is machine learning?", "Explain the capital of France."],
#         "query_time": ["2024-12-01T12:00:00Z", "2024-12-01T12:05:00Z"],
#     }
#     answers = rag_model.batch_generate_answer(batch, retrieval_results)
#     for i, answer in enumerate(answers, 1):
#         print(f"Answer {i}: {answer}")
