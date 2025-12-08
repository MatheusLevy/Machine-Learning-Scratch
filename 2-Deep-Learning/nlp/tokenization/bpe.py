from collections import Counter
from typing import Iterable, Optional
import regex as re
import tiktoken
from bpe_tokenizer import pre_tokenize_rust

IntSeq = tuple[int, ...]
Pair = tuple[int, int]

class BPE():
    def __init__(self, pat_str, target_merges: int=3, vocab_size:int=512, special_tokens: Optional[list[str]]=None):
        self.pat_str = pat_str
        self.regex = re.compile(self.pat_str)
        self.merges: list[tuple[Pair, int]] = []
        self.ranks: dict[Pair, int] = {}    
        self.next_id: int  = 256
        self.target_merges: int = target_merges
        self.vocab_size: int = vocab_size
        self.encoder: dict[IntSeq, int] = {(i,): i for i in range(256)}
        self.decoder: dict[int, IntSeq] = {i: (i,) for i in range(256)}
        self.special_tokens: list[str] = special_tokens or  []
        self.special_token_to_id: dict[str, int] = {}
        self.add_special_token()

    def add_special_token(self) -> None:
        for token_str in self.special_tokens:
            token_int: IntSeq = tuple(token_str.encode("utf-8"))
            if token_int in self.encoder:
                token_id = self.encoder[token_int]
            else:
                token_id = self.next_id
                self.encoder[token_int] = token_id
                self.decoder[token_id] = token_int
                self.next_id += 1
            self.special_token_to_id[token_str] = token_id

    def decode_to_bytes(self, token_ids: list[int]) -> bytes:
            return b"".join(bytes(self.decoder[token_id]) for token_id in token_ids)
    
    def decode_text(self, token_ids: list[int]) -> str:
        return self.decode_to_bytes(token_ids).decode("utf-8", errors="strict")
    
    def pre_tokenize(self, text: str) -> list[str]:
        return pre_tokenize_rust(self.pat_str, text)

    @staticmethod
    def _get_all_pairs(tokens: list[int]) -> Iterable[Pair]:
        for i in range(len(tokens) - 1):
            yield (tokens[i], tokens[i+1])
    
    @staticmethod
    def text_to_ints(text: str) -> list[int]:
        return list(text.encode('utf-8'))
    
    def replace_pair_in_chunk(self, chunk: list[int], pair: Pair) -> list[int]:
        replaced_chunk: list[int] = []
        i: int = 0
        first_part_intseq = self.decoder[pair[0]]
        second_part_intseq = self.decoder[pair[1]]
        merged_int_seq = first_part_intseq + second_part_intseq
        merged_token_id = self.encoder[merged_int_seq]

        def add_current_token() -> None:
            replaced_chunk.append(chunk[i])
        
        def add_merged_token() -> None:
            replaced_chunk.append(merged_token_id)
        
        def pair_found_at_current_position() -> bool:
            return (i + 1 < len(chunk) and 
                chunk[i] == pair[0] and 
                chunk[i + 1] == pair[1])
        
        while i < len(chunk):
            if pair_found_at_current_position():
                add_merged_token()
                i+=2
            else:
                add_current_token()
                i+=1
        
        return replaced_chunk
    
    def _replace_pair(self, chunks: list[list[int]], pair: Pair) -> list[list[int]]:
        return [self.replace_pair_in_chunk(chunk, pair) for chunk in chunks]
    
    def _count_pairs_frequency(self, chunks: list[list[int]]) -> Counter[Pair]:
        def chunk_to_small_for_merging(chunk: list[int]) -> bool:
            return len(chunk) < 2

        global_counter: Counter[Pair] = Counter()
        for chunk in chunks:
            if chunk_to_small_for_merging(chunk): continue
            pairs: zip[int, int] = self._get_all_pairs(chunk)
            global_counter.update(pairs)
        return global_counter


    def _get_most_frequent_pair(self, chunks: list[list[int]]) -> Optional[Pair]:
        def tie_breaker_key(kv: tuple[Pair, int]) -> tuple[int, Pair]:
            pair, freq = kv
            resulting = self.decoder[pair[0]] + self.decoder[pair[1]]
            return (-freq, resulting, pair)

        def hasnt_pairs() -> bool:
            return not global_counter
        
        global_counter: Counter[Pair] = self._count_pairs_frequency(chunks)
        if hasnt_pairs(): return None
        best_pair, _ = min(global_counter.items(), key=lambda kv: tie_breaker_key(kv))
        return best_pair

    def _register_merge(self, pair: Pair, merges_done: int):
        def token_already_exists() -> bool:
            return merged_token in self.encoder

        first_token: IntSeq = self.decoder[pair[0]]
        second_token: IntSeq = self.decoder[pair[1]]
        merged_token: IntSeq = first_token + second_token

        if token_already_exists(): return
        
        self.encoder[merged_token] = self.next_id
        self.decoder[self.next_id] = merged_token

        self.merges.append((pair, self.next_id))
        self.next_id += 1
        self.ranks[pair] = merges_done

    def train(self, corpus: str):
        def not_enough_merges_done() -> bool:
            nonlocal merges_done
            return merges_done < self.target_merges
            
        pre_tokens: list[str] = self.pre_tokenize(corpus)
        chunks: list[list[int]] = [self.text_to_ints(token) for token in pre_tokens]
        max_merges: int = max(0, self.vocab_size - self.next_id)
        self.target_merges: int = min(self.target_merges, max_merges)

        merges_done: int = 0
        while not_enough_merges_done():
            best_pair: Optional[Pair] = self._get_most_frequent_pair(chunks)
            if best_pair is None: break
            self._register_merge(best_pair, merges_done)
            chunks: list[list[int]] = self._replace_pair(chunks, best_pair)
            
            merges_done += 1

        print("Training complete."
              f" Total merges done: {merges_done}."
              f" Vocabulary size: {256 + len(self.merges)}.")
        
    def tokenize(self, text: str) -> list[int]:
        pre_tokens: list[str] = self.pre_tokenize(text)
        chunks: list[list[int]] = [self.text_to_ints(token) for token in pre_tokens]
        replaced_chunks: list[list[int]] = []
        for chunk in chunks:
            while True:
                pairs_in_chunk: set[Pair] = set(self._get_all_pairs(chunk))
                if not pairs_in_chunk:
                    break
                valid_pairs: set[Pair] = pairs_in_chunk.intersection(self.ranks.keys())
                if not valid_pairs:
                    break
                best_pair: Pair = min(valid_pairs, key=lambda pair: self.ranks[pair])
                chunk = self.replace_pair_in_chunk(chunk, best_pair)
            replaced_chunks.append(chunk)
        return [token_id for chunk in replaced_chunks for token_id in chunk]
    
if __name__ == "__main__":
    model = "gpt-5"
    enc = tiktoken.encoding_for_model(model)
    tokenizer = BPE(pat_str=enc._pat_str)
    tokenizer.train(corpus="Helloll, world!")
    print(tokenizer.tokenize("Hello, world!"))