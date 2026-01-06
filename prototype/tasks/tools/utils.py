import os
import traceback
import time
import json
import pickle
import tiktoken
import re, string
from openai import OpenAI
from anthropic import Anthropic

encoding = tiktoken.get_encoding('cl100k_base')


def count_tokens(text):
    # return len(encoding.encode(text, allowed_special={'<|endoftext|>'}))
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Token counting error: {e}")
        return len(encoding.encode(text, allowed_special={'<|endoftext|>'}))


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def F1(answer, key):
    key = normalize_answer(key)
    answer = normalize_answer(answer)
    f1_score = calculate_f1_score(answer, key)
    return f1_score

def EM(answer, key):
    return normalize_answer(answer) == normalize_answer(key)

def calculate_f1_score(predicted_answer, true_answer):
    predicted_set = set(re.split(r'[ -]', predicted_answer))
    true_set = set(re.split(r'[ -]', true_answer))

    if len(predicted_set) == 0:
        return 0

    true_positives = len(predicted_set.intersection(true_set))

    precision = true_positives / len(predicted_set)
    recall = true_positives / len(true_set)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


class APIClient():
    def __init__(self, api, key_path, model, embedding_model):
        assert key_path.endswith(".txt"), "api key path must be a txt file."
        self.api = api
        self.model = model
        self.embedding_model = embedding_model
        if api == "openai":
            self.client = OpenAIClient(key_path, model)
        elif api == "anthropic":
            self.client = AnthropicClient(key_path, model)
        else:
            raise ValueError(f"API {api} not supported, custom implementation required.")

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        return self.client.obtain_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def obtain_embedding(self, input):
        # return self.client.get_embedding(input, self.embedding_model)
        return self.client.obtain_embedding(input, self.embedding_model)


class BaseClient:
    def __init__(self, key_path, model):
        with open(key_path, "r") as f:
            self.key = f.read().strip()
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        response = None
        num_attempts = 0
        while response is None:
            try:
                response = self.send_request(prompt, max_tokens, temperature)
            except Exception as e:
                print(e)
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response

    def send_request(self, prompt, max_tokens, temperature):
        raise NotImplementedError("send_request method must be implemented by subclasses.")


class OpenAIClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model)
        self.client = OpenAI(api_key=self.key, base_url="https://az.gptplus5.com/v1")

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def obtain_embedding(self, text, model):
        embedding = None
        num_attempts = 0
        while embedding is None:
            try:
                embedding = self.get_embedding(text, model)
            except Exception as e:
                print(f"Embedding Error: {e}")
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        return embedding


class AnthropicClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model)
        self.client = Anthropic(api_key=self.key)

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
