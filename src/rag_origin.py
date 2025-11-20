import json
import os
import time
from collections import deque
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, Any, List, Set, Tuple

load_dotenv()

# PROMPT = """
# # Question:
# {{QUESTION}}
#
# # Context:
# {{CONTEXT}}
#
# # Short answer:
# """

# PROMPT = """
# # Question:
# {{QUESTION}}
#
# # Context:
# {{CONTEXT}}
#
# # Question Remind:
# {{QUESTIONREMIND}}
#
# # Short answer:
# """

PROMPT = """
# Context:
{{CONTEXT}}

# Question:
{{QUESTION}}

# Short answer:
"""

class RAGManager:
    def __init__(self, data_path="dataset/locomo10_qa_test.json", chunk_size=500, k=1, on_dgx=False):
        self.model = os.getenv("MODEL")
        GPT_OSS_BASE_URL = os.getenv("GPT_OSS_BASE_URL", "http://115.145.179.241:8000/v1")
        GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY", "EMPTY")
        if on_dgx:
            self.client = OpenAI(
                base_url=GPT_OSS_BASE_URL,
                api_key=GPT_OSS_API_KEY, # EMPTY
            )
        else:
            self.client = OpenAI()
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k
    # def __init__(self, data_path="dataset/locomo10_qa_test.json", chunk_size=500, k=1):
    #     self.model = os.getenv("MODEL")
    #     self.client = OpenAI()
    #     self.data_path = data_path
    #     self.chunk_size = chunk_size
    #     self.k = k

    def generate_response(self, question, context, qnum):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question, QUESTIONREMIND=question, QNUM=qnum)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                #response = self.client.chat.completions.create(
                stream =    self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    # temperature=0,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                t_first = None  # 첫 토큰까지 걸린 시간
                gaps = deque()  # 토큰‑간 간격(초)
                t_prev = None
                answer_parts = []
                usage = None  # 마지막 청크에서 채워짐

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        answer_parts.append(chunk.choices[0].delta.content)

                    now = time.time()

                    if t_first is None:  # 첫 루프 → t_first
                        t_first = now - t1
                    elif t_prev is not None:  # 두 번째 토큰부터는 간격 기록
                        gaps.append(now - t_prev)

                    t_prev = now

                    # usage 정보는 마지막 extra‑chunk에 들어옴
                    if chunk.usage is not None:
                        usage = chunk.usage
                    # print(chunk)

                t_total = now - t1  # 전체 소요 시간
                tpot_avg = (sum(gaps) / len(gaps)) if gaps else 0.0

                return "".join(answer_parts).strip(), t_first, tpot_avg, t_total ,usage

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def generate_response_origin(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question, QUESTIONREMIND=question)

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    # temperature=0,
                )
                t2 = time.time()

                # Token Usage
                usage = response.usage
                return response.choices[0].message.content.strip(), t2 - t1, usage
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"

        return cleaned_chat_history

    def calculate_embedding(self, document):
        response = self.client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=document)
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query: The query string
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
        """
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]

        # Get indices of top-k most similar chunks
        if k == 1:
            # Original behavior - just get the most similar chunk
            top_indices = [np.argmax(similarities)]
        else:
            # Get indices of top-k chunks
            top_indices = np.argsort(similarities)[-k:][::-1]

        # Combine the top-k chunks
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for more accurate token counting
        """

        documents = self.clean_chat_history(chat_history)

        # # Encode the document
        # encoding = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))
        # tokens = encoding.encode(documents)
        # chunks = []
        # # Split into chunks based on token count
        # for i in range(0, len(tokens), 8192):
        #     chunk_tokens = tokens[i: i + chunk_size]
        #     chunk = encoding.decode(chunk_tokens)
        #     chunks.append(chunk)
        # return chunks[0], []

        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Get the encoding for the model
        encoding = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))

        # Encode the document
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        embeddings = []
        for chunk in chunks:
            embedding = self.calculate_embedding(chunk)
            embeddings.append(embedding)

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            # with open("results/conv_context.jsonl", "a") as f:  # ← "a" = append
            #     json.dump({"conv_idx": key, "context": chunks[0]}, f)
            #     f.write("\n")  # 줄 구분

            qnum = 0
            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
                response, time_prefill, time_decode_avg, response_time, usage = self.generate_response(question, context, qnum)
                # usage

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        # "context": context,
                        "response": response,
                        # "search_time": search_time,
                        "response_time": response_time,
                        "prefill_time": time_prefill,
                        "decode_time_avg": time_decode_avg,
                        "total_tokens": usage.total_tokens,
                        "prompt_tokens": usage.prompt_tokens,
                        "prompt_cached_tokens": usage.prompt_tokens_details.cached_tokens,  # prompt cache
                        "completion_tokens": usage.completion_tokens,
                        # "completion_reasoning_tokens": usage.completion_tokens_details.reasoning_tokens,
                        # "completion_accept_tokens" : usage.completion_tokens_details.accepted_prediction_tokens,
                        # "completion_reject_tokens" : usage.completion_tokens_details.rejected_prediction_tokens,
                    }
                )
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)
                # response, response_time, usage = self.generate_response(question, context)
                #
                # FINAL_RESULTS[key].append(
                #     {
                #         "question": question,
                #         "answer": answer,
                #         "category": category,
                #         #"context": context,
                #         "response": response,
                #         #"search_time": search_time,
                #         "response_time": response_time,
                #         "total_tokens": usage.total_tokens,
                #         "prompt_tokens": usage.prompt_tokens,
                #         "completion_tokens": usage.completion_tokens,
                #         "completion_reasoning_tokens" : usage.completion_tokens_details.reasoning_tokens,
                #     }
                # )
                # with open(output_file_path, "w+") as f:
                #     json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
